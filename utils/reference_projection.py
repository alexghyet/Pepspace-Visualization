"""Persist fitted UMAP models and transform queries without refitting references."""

from io import BytesIO
from time import perf_counter

import joblib
import numpy as np

from utils.dataset import validate_embeddings
from utils.embedding_runtime import dependency_versions, memory


@memory.cache
def fit_reference(reference, parameters, versions):
    import umap.umap_ as umap

    kwargs = dict(parameters)
    if len(reference) <= kwargs["n_components"] + 1:
        kwargs["init"] = "random"
    start = perf_counter()
    reducer = umap.UMAP(**kwargs).fit(reference)
    coordinates = reducer.embedding_.copy()
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP produced nonfinite reference coordinates. Try different projection settings.")
    return reducer, coordinates, perf_counter() - start


@memory.cache
def transform_query(reference, query, parameters, versions):
    reducer, _, _ = fit_reference(reference, parameters, versions)
    coordinates = reducer.transform(query)
    if not np.isfinite(coordinates).all():
        raise ValueError("Some new peptides could not be placed in this reference map. Check the vectors or projection settings.")
    return coordinates


def reference_parameters(n_components, n_neighbors, min_dist, metric, seed, method, **unused):
    if method != "UMAP":
        raise ValueError("Reference overlays require UMAP; densMAP cannot transform unseen points.")
    return dict(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                metric=metric, random_state=seed, transform_seed=seed, n_jobs=1)


def get_reference_projection(reference, query, **settings):
    reference = validate_embeddings(reference)
    query = validate_embeddings(query, min_rows=1)
    if reference.shape[1] != query.shape[1]:
        raise ValueError("Reference and new embeddings must have the same number of features.")
    parameters = reference_parameters(**settings)
    versions = dependency_versions()
    args = (reference, parameters, versions)
    cached = fit_reference.check_call_in_cache(*args) and transform_query.check_call_in_cache(reference, query, parameters, versions)
    start = perf_counter()
    _, coordinates, fit_seconds = fit_reference(*args)
    new_coordinates = transform_query(reference, query, parameters, versions)
    return np.vstack([coordinates, new_coordinates]), fit_seconds, perf_counter() - start, cached


def fit_display(reference_coordinates):
    """Fit PCA and globe normalization using only reference coordinates."""
    reference = np.asarray(reference_coordinates, dtype=np.float64)
    if reference.shape[1] > 3:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3, svd_solver="full").fit(reference)
        mean, components = pca.mean_, pca.components_
    else:
        mean, components = reference.mean(axis=0), np.eye(3)
    centered = (reference - mean) @ components.T
    extent = np.linalg.norm(centered, axis=1).max()
    return {"mean": mean, "components": components, "scale": .9 / extent if extent > 0 else 1.}


def apply_display(coordinates, transform, layout="Inside globe"):
    centered = (np.asarray(coordinates) - transform["mean"]) @ transform["components"].T
    if layout == "Inside globe":
        return centered * transform["scale"]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)


def reference_bundle(reference, dimensions, settings, provenance):
    """Export only locally fitted models; the app never loads uploaded pickles."""
    bundle = {"schema_version": 1, "provenance": provenance, "versions": dependency_versions(), "models": {}}
    for dimension in dimensions:
        params = reference_parameters(n_components=dimension, **settings)
        reducer, coordinates, _ = fit_reference(reference, params, dependency_versions())
        bundle["models"][dimension] = {
            "reducer": reducer, "coordinates": coordinates,
            "display": fit_display(coordinates) if dimension >= 3 else None,
        }
    stream = BytesIO()
    joblib.dump(bundle, stream, compress=3)
    return stream.getvalue()

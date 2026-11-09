"""Exact, persistent projection caching shared by the app and precomputation."""

from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from joblib import Memory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
memory = Memory(PROJECT_ROOT / ".cache" / "projections", verbose=0)


@lru_cache(maxsize=1)
def dependency_versions():
    # Avoid reusing a projection made by a different scientific-library version.
    return tuple(
        (name, version(name))
        for name in ("numpy", "umap-learn", "numba", "scikit-learn", "pynndescent")
    )


@memory.cache
def _fit_umap(X, parameters, library_versions):
    # Imported only on a cache miss; a warm launch avoids UMAP/Numba startup.
    import umap.umap_ as umap

    kwargs = dict(parameters)
    # Spectral initialization requires more samples than output dimensions + 1.
    if len(X) <= kwargs["n_components"] + 1:
        kwargs["init"] = "random"
    start = perf_counter()
    embedding = umap.UMAP(**kwargs).fit_transform(X)
    return embedding, perf_counter() - start


def get_umap_projection(
    X, n_neighbors, min_dist, n_components, metric, seed, method,
    dens_lambda=None, dens_frac=None, dens_var_shift=None,
):
    parameters = dict(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        metric=metric, random_state=seed,
    )
    if method == "densMAP":
        parameters.update(
            densmap=True, dens_lambda=dens_lambda,
            dens_frac=dens_frac, dens_var_shift=dens_var_shift,
        )
    args = (X, parameters, dependency_versions())
    cache_hit = _fit_umap.check_call_in_cache(*args)
    start = perf_counter()
    embedding, fit_seconds = _fit_umap(*args)
    return embedding, fit_seconds, perf_counter() - start, cache_hit


@memory.cache
def _exact_trustworthiness(X, coordinates, n_neighbors, metric, library_versions):
    from sklearn.manifold import trustworthiness

    return float(trustworthiness(X, coordinates, n_neighbors=n_neighbors, metric=metric))


def get_trustworthiness(X, coordinates, n_neighbors, metric="euclidean"):
    """Cache the full-data score, retaining the app's original Euclidean metric."""
    return _exact_trustworthiness(
        X, coordinates, n_neighbors, metric, dependency_versions()
    )

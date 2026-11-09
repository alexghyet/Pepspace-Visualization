"""Display coordinates for a globe; these never modify the source embeddings."""

import numpy as np


def centered_3d_coordinates(X):
    """Center 3D input, or use deterministic PCA to display higher dimensions.

    PCA from >3 dimensions is a lossy display projection. Centering 3D input
    preserves every pairwise distance and avoids an origin-dependent sphere cap.
    """
    coordinates = np.asarray(X, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError("Expected an array with at least three dimensions.")
    if len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError("Expected at least three finite coordinate rows.")
    if coordinates.shape[1] > 3:
        from sklearn.decomposition import PCA

        return PCA(n_components=3, svd_solver="full").fit_transform(coordinates)
    return coordinates - coordinates.mean(axis=0, keepdims=True)


def fit_embeddings_in_sphere(X, radius=0.9):
    """Fit inside a globe with ONE scale factor, preserving 3D geometry.

    This preserves all relative 3D distances, angles, and nearest neighbors.
    It does not claim to preserve all distances in the original embedding space.
    """
    if not 0 < radius <= 1:
        raise ValueError("The radius must be in (0, 1].")
    coordinates = centered_3d_coordinates(X)
    extent = np.linalg.norm(coordinates, axis=1).max()
    return coordinates * (radius / extent) if extent > 0 else coordinates


def project_embeddings_to_sphere(X):
    """Center and radially project onto a unit sphere surface.

    Radial projection changes distances and may change neighborhoods. A point
    exactly at the centroid has no direction and remains at the origin.
    """
    coordinates = centered_3d_coordinates(X)
    norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
    return np.divide(
        coordinates, norms, out=np.zeros_like(coordinates), where=norms > 0
    )

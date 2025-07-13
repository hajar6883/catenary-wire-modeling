# utils.py
"""Utility functions for wire_cluster package."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Projection helpers
# -----------------------------------------------------------------------------

def select_axes_2d(points3d: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Project a 3‑D array to 2‑D by selecting two coordinate axes."""
    return points3d[:, axes]


# PCA helpers
# -----------------------------------------------------------------------------


def pca_project_clustering(points2d: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Rotate a 2‑D point set so that PC1 aligns with the direction of maximum spread.

    This is often useful when wires are diagonal in the chosen plane – after PCA,
    the second component (PC2) captures the inter‑wire separation.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(points2d)


def pca_fit_plane(points3d: np.ndarray, n_components: int = 2):
    """
    Fit a PCA model to a 3-D point cloud and return:
      • the 2-D point cloud in the PCA plane
      • the fitted PCA object (needed for back-projection)

    """
    pca = PCA(n_components=n_components)
    points2d = pca.fit_transform(points3d)
    return points2d, pca





def pca_back_project(points2d: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Inverse-transform 2-D points from the PCA plane back into 3-D space"""
    return pca.inverse_transform(points2d)


# 1‑D clustering helper
# -----------------------------------------------------------------------------

def cluster_1d(values_1d: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN on a single‑dimensional array of values.

    The input can be 1‑D; we reshape to (N, 1) for sklearn compatibility.
    Returns the label array of length N.
    """
    if values_1d.ndim == 1:
        values_1d = values_1d.reshape(-1, 1)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(values_1d)

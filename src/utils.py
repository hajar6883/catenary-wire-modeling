# utils.py
"""Utility functions for wire_cluster package."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Projection helpers
# -----------------------------------------------------------------------------

def project_to_2d(points3d: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Project a 3‑D array to 2‑D by selecting two coordinate axes.

    Parameters
    ----------
    points3d : np.ndarray
        Array of shape (N, 3).
    axes : tuple[int, int]
        Which axes to keep. Default `(0, 1)` → X‑Y plane.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2).
    """
    return points3d[:, axes]


# PCA helpers
# -----------------------------------------------------------------------------

def run_pca_projection(points2d: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Rotate a 2‑D point set so that PC1 aligns with the direction of maximum spread.

    This is often useful when wires are diagonal in the chosen plane – after PCA,
    the second component (PC2) captures the inter‑wire separation.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(points2d)


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

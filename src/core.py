# core.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
from scipy.optimize import curve_fit


from .utils import (
    select_axes_2d,
    pca_project_clustering,
    cluster_1d)

from .models import (
    catenary,               # for reuse if needed
    fit_catenary_2d,        # fitting logic
    fit_catenary_wire # full 3D workflow
)
from .presets import DBSCAN_PARAMS, HDBSCAN_PARAMS


def is_flat_z(points3d, threshold=2.0):
    """difference between the value just below the top 5% and the bottom 5% (ignores very high/low outliers) 
    to tell tells us the main vertical range of the wires 
    Returns True if it is FLAT or False if each selection of wire sis STACKED
    """
    z_range = np.percentile(points3d[:, 2], 95) - np.percentile(points3d[:, 2], 5) 
    return z_range < threshold


def simple_clustering(points3d):
    """
    Assumes wires are roughly on the same Z level.
    Projects to X-Y, applies PCA, then DBSCAN on PCA axis 2.
    """
    points2d = select_axes_2d(points3d, axes=[0, 1])  # X-Y projection
    points2d = pca_project_clustering(points2d)
    labels = cluster_1d(points2d[:, 1], **DBSCAN_PARAMS)
    return labels


def hierarchical_clustering(points3d):
    """
    Handles vertically stacked wires using a two-stage approach:
        1. Use HDBSCAN on Z or YZ to find major clusters (upper/lower layers).
        2. Inside each, run PCA + DBSCAN on orthogonal axis to split wires.
    """

    major_clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    major_labels = major_clusterer.fit_predict(points3d[:, [1, 2]])  # Y-Z projection

    final_labels = np.full(len(points3d), -1)
    wire_counter = 0

    for major_label in np.unique(major_labels):
        if major_label == -1:
            continue

        mask = major_labels == major_label
        cluster_points = points3d[mask]

        points2d = select_axes_2d(cluster_points, axes=[0, 1])
        points2d = pca_project_clustering(points2d)
        wire_labels = cluster_1d(points2d[:, 1], **DBSCAN_PARAMS)

        final_labels[mask] = wire_labels + wire_counter
        wire_counter += wire_labels.max() + 1 if wire_labels.max() != -1 else 0

    return final_labels



def cluster_wires(points3d: np.ndarray) -> np.ndarray:
    """
    Main interface: determines wire structure and applies appropriate clustering pipeline.
    Requires: NumPy array of shape (N, 3), where columns are [x, y, z].
    """
    if not isinstance(points3d, np.ndarray):
        raise TypeError("Input must be a NumPy array (use .values if DataFrame)")

    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {points3d.shape}")

    # Route to clustering
    if is_flat_z(points3d):
        return simple_clustering(points3d)
    else:
        return hierarchical_clustering(points3d)
    



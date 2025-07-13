#models.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .utils import (
    pca_fit_plane,
    pca_back_project,
)



def catenary(x, x0, y0, c):
        return y0 + c * (np.cosh((x - x0) / c) - 1)


def fit_catenary_2d(points_2d, n_fit_points=500, p0=None):
   
    # Sort by x for stability
    sorted_idx = np.argsort(points_2d[:, 0])
    points_2d = points_2d[sorted_idx]

    x_data, y_data = points_2d[:, 0], points_2d[:, 1]
    # Use default p0 if not provided
    if p0 is None:
        p0 = [np.mean(x_data), np.min(y_data), 1.0]



    try:
        params, _ = curve_fit(catenary, x_data, y_data, p0=p0)
    except RuntimeError:
        return None, None

    # Smooth curve for visualisation
    x_fit = np.linspace(x_data.min(), x_data.max(), n_fit_points)
    y_fit = catenary(x_fit, *params)
    curve_2d = np.column_stack((x_fit, y_fit))

    return curve_2d, params


def fit_catenary_wire(points3d):
    """
    Full wire fit:
    1) PCA → 2-D
    2) catenary fit
    3) back-project to 3-D
    Returns (curve3d, params) or None on failure.
    """
    pts_2d, pca = pca_fit_plane(points3d)
    curve_2d, params = fit_catenary_2d(pts_2d)

    if curve_2d is None:
        return None
    
    curve_3d = pca_back_project(curve_2d, pca)
    # RMSE eval
    tree = cKDTree(curve_3d)
    dists, _ = tree.query(points3d)
    rmse = np.sqrt(np.mean(dists**2))
    return curve_3d, params, rmse



# def fit_all_wires(points3d, labels, verbose=True):
#     fits_to_plot = []
#     for wire_id in sorted(set(labels)):
#         if wire_id == -1:
#             continue
#         wire_points = points3d[labels == wire_id]
#         result = fit_catenary_wire(wire_points)
#         if result is None:
#             if verbose:
#                 print(f"[!] Wire {wire_id}: Fit failed.")
#             continue

#         curve3d, params, rmse = result
#         x0, y0, c = params
#         if verbose:
#             print(f"[✓] Wire {wire_id}: x0={x0:.2f}, y0={y0:.2f}, c={c:.2f}, RMSE={rmse:.4f}")

#         fits_to_plot.append((wire_id, wire_points, curve3d))

#     return fits_to_plot









# attempt to use optimization in 3D (instead of PCA projection) 
# -----------------------------------------------------------------------------------------

def catenary_model_3d(t, x0, y0, z0, dx, dz, c):
    """
    Parametric catenary curve in 3D space.
    - dx, dz: define the direction of the wire in X and Z
    """
    x = x0 + t * dx
    y = y0 + c * (np.cosh(t / c) - 1)
    z = z0 + t * dz
    return np.vstack((x, y, z)).T

def fit_catenary_direct_3d(points3d, n_fit_points=500):
    """
    Fit a 3D catenary curve directly to 3D points (no PCA).
    Returns fitted curve3d and parameters.
    """
    # Center the data for stability
    centroid = points3d.mean(axis=0)
    centered = points3d - centroid

    # Fit direction vector using PCA (only for estimating tangent)
    u, s, vh = np.linalg.svd(centered)
    direction = vh[0]  # principal direction
    dx, _, dz = direction

    # Parametrize t for input points by projecting onto the direction
    t_values = centered @ direction

    x_data, y_data, z_data = centered.T

    def loss(params):
        x0, y0, z0, c = params
        curve = catenary_model_3d(t_values, x0, y0, z0, dx, dz, c)
        residual = curve - centered
        return np.mean(np.sum(residual**2, axis=1))  # MSE

    # Init params
    p0 = [0, np.min(y_data), 0, 1.0]

    res = minimize(loss, p0, method="L-BFGS-B", bounds=[(-10, 10), (-10, 10), (-10, 10), (0.01, 1000)])

    if not res.success:
        return None

    x0, y0, z0, c = res.x

    # Rebuild fitted curve
    t_fit = np.linspace(t_values.min(), t_values.max(), n_fit_points)
    curve_centered = catenary_model_3d(t_fit, x0, y0, z0, dx, dz, c)

    # Add back centroid to restore original position
    curve3d = curve_centered + centroid

    return curve3d, (x0 + centroid[0], y0 + centroid[1], z0 + centroid[2], c)




def fit_all_wires(points3d, labels, fit_fn, verbose=True):
    """
    Generic fitter that works with either 2D or 3D catenary model functions.

    Parameters
    ----------
    points3d : np.ndarray
        Input point cloud (Nx3)
    labels : np.ndarray
        Cluster labels for each point
    fit_fn : callable
        Fitting function to apply to each cluster. Must return (curve3d, params[, rmse])
    verbose : bool
        Whether to print info for each wire

    Returns
    -------
    fits_to_plot : list of (wire_id, wire_points, curve3d)
    """
    fits_to_plot = []

    for wire_id in sorted(set(labels)):
        if wire_id == -1:
            continue

        wire_points = points3d[labels == wire_id]
        result = fit_fn(wire_points)

        if result is None:
            if verbose:
                print(f"[!] Wire {wire_id}: Fit failed.")
            continue

        # unpack based on output size
        if len(result) == 2:
            curve3d, params = result
            rmse = None
        else:
            curve3d, params, rmse = result

        if verbose:
            msg = f"[✓] Wire {wire_id}: " + ", ".join([f"{k}={v:.2f}" for k, v in zip("x0 y0 c".split(), params[:3])])
            if rmse is not None:
                msg += f", RMSE={rmse:.4f}"
            print(msg)

        fits_to_plot.append((wire_id, wire_points, curve3d))

    return fits_to_plot

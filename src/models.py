from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import numpy as np



def catenary(x, x0, y0, c):
        return y0 + c * (np.cosh((x - x0) / c) - 1)


def project_to_2d(points3d):
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points3d)
    return points_2d, pca


def fit_catenary_2d(points_2d, n_fit_points=500):
    sorted_idx = np.argsort(points_2d[:, 0])
    points_2d = points_2d[sorted_idx]
    x_data = points_2d[:, 0]
    y_data = points_2d[:, 1]

    init_params = [np.mean(x_data), np.min(y_data), 1.0]
    try:
        params, _ = curve_fit(catenary, x_data, y_data, p0=init_params)
    except RuntimeError:
        return None, None

    x_fit = np.linspace(np.min(x_data), np.max(x_data), n_fit_points)
    y_fit = catenary(x_fit, *params)
    points_fit_2d = np.column_stack((x_fit, y_fit))
    
    return points_fit_2d, params

def project_to_3d(points2d, pca):
    return pca.inverse_transform(points2d)


def fit_catenary_2d(points_2d, n_fit_points=500):
   
    # Sort by x for stability
    sorted_idx = np.argsort(points_2d[:, 0])
    points_2d = points_2d[sorted_idx]

    x_data, y_data = points_2d[:, 0], points_2d[:, 1]
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


def fit_catenary_to_cluster(points3d):
    """
    Full wire fit:
    1) PCA → 2-D
    2) catenary fit
    3) back-project to 3-D
    Returns (curve3d, params) or None on failure.
    """
    pts_2d, pca = project_to_2d(points3d)
    curve_2d, params = fit_catenary_2d(pts_2d)

    if curve_2d is None:
        return None

    curve_3d = project_to_3d(curve_2d, pca)
    return curve_3d, params




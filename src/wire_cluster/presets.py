# presets.py
"""
Default hyperparameters for clustering algorithms
"""

DBSCAN_PARAMS = {
    'eps': 0.05,
    'min_samples': 5
}

HDBSCAN_PARAMS = {
    'min_cluster_size': 20,
    'min_samples': 10,
    'cluster_selection_epsilon': 0.0, 
    'cluster_selection_method': 'eom'
}

from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def cluster_points_dbscan(points3d, eps=.765, min_samples=8, z_scale=1.0):
    """
    Cluster 3-D points with DBSCAN.
    Optional z-scaling (helps when Z range is tiny vs X/Y).
    Returns an array of cluster labels (-1 = noise).
    (Default params defined are fine turned for the 'lidar_cable_points_easy' data points)
    """
    scaled = points3d.copy()
    scaled[:, 2] *= z_scale
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    return labels




def plot_clusters_3d(points3d, labels, true_range = True, title="Clustered Point Cloud" ):
    """
    Visualizes 3D clustered points using Plotly.
    Points labeled -1 (noise) are shown in gray.
    """
    fig = go.Figure()
    unique_labels = np.unique(labels)

    colors = px.colors.qualitative.Bold + px.colors.qualitative.Set1

    for i, label in enumerate(unique_labels):
        cluster_points = points3d[labels == label]
        color = 'gray' if label == -1 else colors[i % len(colors)]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=color),
            name=f'Cluster {label}' if label != -1 else 'Noise'
        ))
    scene_config = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
    if true_range:
        scene_config['aspectmode'] = 'data'

    fig.update_layout(
        title=title,
        scene=scene_config,

        # width=800,
        # height=600,
        showlegend=True
    )
    fig.show()

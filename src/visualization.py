from matplotlib import cm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# def plot_clusters(points3d, labels=None):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_facecolor("#f8f9fa")  # light background
#     fig.patch.set_facecolor("#f8f9fa")

#     x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]

#     if labels is not None:
#         labels = labels.astype(str)
#         unique_labels = np.unique(labels)
#         cmap = plt.get_cmap("tab10")  # You can try "Set2", "viridis", etc.

#         for i, lbl in enumerate(unique_labels):
#             mask = labels == lbl
#             ax.scatter(
#                 x[mask], y[mask], z[mask],
#                 label=f"Cluster {lbl}",
#                 s=10,  # marker size
#                 alpha=0.7,
#                 color=cmap(i % 10)
#             )
#         ax.legend(loc="upper right")
#         ax.set_title("Clustered Wires (3D View)", fontsize=14)
#     else:
#         ax.scatter(x, y, z, s=10, alpha=0.7, color="dodgerblue")
#         ax.set_title("Raw Point Cloud (3D View)", fontsize=14)

#     ax.set_xlabel("X", fontsize=12)
#     ax.set_ylabel("Y", fontsize=12)
#     ax.set_zlabel("Z", fontsize=12)

#     plt.tight_layout()
#     plt.show()


def plot_clusters_plotly(points3d, labels=None, output_html="cluster_plot.html"):
    """
    Plot clustered 3D wire points using Plotly and save to HTML.

    Args:
        points3d (np.ndarray): N x 3 array of 3D points.
        labels (np.ndarray): Optional cluster labels (same length as points3d).
        output_html (str): Filename to save interactive HTML plot.
    """
    df = pd.DataFrame(points3d, columns=["X", "Y", "Z"])
    df["Cluster"] = labels.astype(str) if labels is not None else "Raw"

    fig = px.scatter_3d(
        df,
        x="X", y="Y", z="Z",
        color="Cluster",
        title="Wire Clusters (3D View)",
        opacity=0.7,
        height=800,
        width=1000
    )

    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="#f8f9fa"
        ),
        legend_title="Cluster ID",
        margin=dict(l=0, r=0, b=0, t=40)
    )

 
    fig.write_html(output_html, auto_open=True)
    print(f"-> 3D cluster plot saved to: {output_html}")



# def plot_fitted_wires_all(fits, max_cols=4):
#     """
#     Plot original + fitted catenary curves for all wires in a single figure with subplots.

#     Parameters
#     ----------
#     fits : list of tuples
#         Each tuple is (wire_id, wire_points, curve3d)
#     max_cols : int
#         Max number of subplot columns (defaults to 4)
#     """
#     n = len(fits)
#     cols = min(n, max_cols)
#     rows = math.ceil(n / cols)

#     fig = plt.figure(figsize=(5 * cols, 5 * rows))
#     fig.patch.set_facecolor("#f8f9fa")

#     for i, (wire_id, wire_points, curve3d) in enumerate(fits):
#         ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
#         ax.set_facecolor("#f8f9fa")

#         # Original points
#         ax.scatter(*wire_points.T, s=3, alpha=0.6, label="Original", color="steelblue")
#         # Fitted curve
#         ax.plot(*curve3d.T, color="crimson", linewidth=2, label="Fitted Catenary")

#         ax.set_title(f"Wire {wire_id}", fontsize=12)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.legend()

#     plt.tight_layout()
#     plt.show()




def plot_all_wires_plotly(all_wire_data, output_path="fitted_wires_plot.html"):
    """
    all_wire_data: list of tuples (wire_id, wire_points, curve3d)
    Plots all wires and their fitted catenaries in a single 3D Plotly figure.
    """
    fig = go.Figure()

    for wire_id, wire_points, curve3d in all_wire_data:
        fig.add_trace(go.Scatter3d(
            x=wire_points[:, 0],
            y=wire_points[:, 1],
            z=wire_points[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.6),
            name=f"Wire {wire_id} Points"
        ))
        fig.add_trace(go.Scatter3d(
            x=curve3d[:, 0],
            y=curve3d[:, 1],
            z=curve3d[:, 2],
            mode='lines',
            line=dict(width=4),
            name=f"Wire {wire_id} Fit"
        ))

    fig.update_layout(
        title="All Wires and Fitted Catenaries",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        template="plotly_white"
    )

    fig.write_html(output_path, auto_open=True)

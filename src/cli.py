import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from src.core import cluster_wires
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_cable_points(path):
    if path.endswith(".npy"):
        return np.load(path)
    
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.values
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.values
    
    
    else:
        raise ValueError("Unsupported file format. Use .npy, .csv, or .parquet")


# def plot_clusters(points3d, labels=None):
    
#     df = pd.DataFrame(points3d, columns=["x", "y", "z"])

#     if labels is not None:
#         df["label"] = labels.astype(str)
#         fig = px.scatter_3d(df, x="x", y="y", z="z", color="label",
#                             title="Clustered Wires (3D View)", opacity=0.7)
#     else:
#         fig = px.scatter_3d(df, x="x", y="y", z="z",
#                             title="Raw Point Cloud (3D View)", opacity=0.7)

#     fig.show()



def plot_clusters(points3d, labels=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8f9fa")  # light background
    fig.patch.set_facecolor("#f8f9fa")

    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]

    if labels is not None:
        labels = labels.astype(str)
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10")  # You can try "Set2", "viridis", etc.

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                x[mask], y[mask], z[mask],
                label=f"Cluster {lbl}",
                s=10,  # marker size
                alpha=0.7,
                color=cmap(i % 10)
            )
        ax.legend(loc="upper right")
        ax.set_title("Clustered Wires (3D View)", fontsize=14)
    else:
        ax.scatter(x, y, z, s=10, alpha=0.7, color="dodgerblue")
        ax.set_title("Raw Point Cloud (3D View)", fontsize=14)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    plt.tight_layout()
    plt.show()





def main():
    parser = argparse.ArgumentParser(description="Wire clustering CLI tool")
    parser.add_argument("--input", required=True, help="Path to .npy, .csv or .parquet file with 3D points")
    parser.add_argument("--plot", action="store_true", help="Visualize 3D clusters after running")

    args = parser.parse_args()
    
    data_path = os.path.join("data", args.input) if not os.path.exists(args.input) else args.input

    points3d = load_cable_points(data_path)
    labels = cluster_wires(points3d)

    print("Clustering complete. Unique labels:", np.unique(labels))

    if args.plot:
        plot_clusters(points3d, labels)


if __name__ == "__main__":
    main()

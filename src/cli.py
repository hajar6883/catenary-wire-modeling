import argparse
import numpy as np
import pandas as pd
import os 

from src.wire_segmentation import cluster_wires
from src.catenary_models import fit_all_wires , fit_catenary_wire

from src.visualization import plot_all_wires_plotly,  plot_clusters_plotly





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




def main():
    parser = argparse.ArgumentParser(description="Wire modeling CLI")
    parser.add_argument("--input", required=True, help="Path to .npy, .csv or .parquet file with 3D points")
    parser.add_argument("--cluster", action="store_true", help="Cluster and visualize segmented wires")
    parser.add_argument("--fit", action="store_true", help="Fit 2-D-plane catenary on each wire (PCA→2-D)")
    parser.add_argument("--plot", action="store_true", help="Produce Plotly HTML visualisations for cluster / fits")
    args = parser.parse_args()

    data_path = os.path.join("data", args.input) if not os.path.exists(args.input) else args.input
    points3d = load_cable_points(data_path)

    if args.cluster or args.fit:
        labels = cluster_wires(points3d)
        print("Clustering complete. Unique labels:", np.unique(labels))

    if args.cluster and not args.fit:
        if args.plot:
            plot_clusters_plotly(points3d, labels)
        else:
            print("Clustering done. Use --plot to get HTML output.")

    if args.fit:
        fits_to_plot = fit_all_wires(points3d, labels, fit_fn=fit_catenary_wire, verbose=True)
        if args.plot and fits_to_plot:
            plot_all_wires_plotly(fits_to_plot, output_path="fitted_wires_2d.html")

    if not (args.cluster or args.fit):
        print("Nothing to do. Use --cluster and/or --fit.")


if __name__ == "__main__":
    main()

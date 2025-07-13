import argparse
import numpy as np
import pandas as pd

from src.core import cluster_wires
from src.models import fit_catenary_wire
import os 

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
    parser.add_argument("--fit", action="store_true", help="Fit catenary curves on each clustered wire and optionally plot 3D fits")
    parser.add_argument("--plot", action="store_true", help="Enable visualization for either clustering or fitting")
    args = parser.parse_args()

    # Resolve input path
    data_path = os.path.join("data", args.input) if not os.path.exists(args.input) else args.input
    points3d = load_cable_points(data_path)

    # Run clustering 
    if args.cluster or args.fit:
        labels = cluster_wires(points3d)
        print("Clustering complete. Unique labels:", np.unique(labels))

    # -- Case 1: Only clustering visualization --
    if args.cluster and not args.fit:
        if args.plot:
            plot_clusters_plotly(points3d, labels)
        else:
            print("Clustering done. Use --plot to visualize.")

    # -- Case 2: Fitting (always includes clustering first) --
    if args.fit:
        
        fits_to_plot = []

        for wire_id in sorted(set(labels)):
            if wire_id == -1:
                continue

            wire_points = points3d[labels == wire_id]
            result = fit_catenary_wire(wire_points)

            if result is None:
                print(f"[!] Wire {wire_id}: Fit failed.")
                continue

            curve3d, params = result
            x0, y0, c = params
            print(f"[✓] Wire {wire_id}: x0={x0:.2f}, y0={y0:.2f}, c={c:.2f}")

            if args.plot:
                fits_to_plot.append((wire_id, wire_points, curve3d))

        # After the loop
        if args.plot and fits_to_plot:
            # plot_fitted_wires_all(fits_to_plot)
            plot_all_wires_plotly(fits_to_plot)

        

    # -- Case 3: Neither clustering nor fitting
    if not args.cluster and not args.fit:
        print("Nothing to do. Use --cluster and/or --fit.")



if __name__ == "__main__":
    main()

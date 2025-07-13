# catenary-wire-modeling
A Python package to detect and model overhead electrical wires from LiDAR point cloud data. 

RMSE was estimated by computing nearest-point distances between the original 3D wire points and a dense sample of the fitted catenary curve.



# LiDAR Wire Detection and Catenary Modeling

This project addresses a data science task, involving the detection and modeling of overhead electricity wires using drone-based LiDAR point cloud data. The goal is to fit 3D catenary curves to individual wires within a point cloud.

---

## Project Overview

Given .parquet LiDAR datasets, the pipeline:
- Segments point clouds to identify individual wires.
- Clusters the points belonging to each wire.
- Fits a best-fit 3D **catenary curve** to each cluster using nonlinear optimization.

This helps power grid operators visualize and analyze the geometry of suspended wires efficiently.

---

## Project Structure

```bash
├── data/                      # Contains the input .parquet LiDAR files
│   └── lidar_cable_points_*.parquet
│
├── src/                       # All source code
│   ├── tests/                 # Unit tests for each module
│   ├── catenary_models.py     # Catenary fitting logic
│   ├── cli.py                 # Command-line interface for running the pipeline
│   ├── presets.py             # Preset configs for clustering/fitting
│   ├── utils.py               # General-purpose utilities
│   ├── visualization.py       # 3D visualizations using Plotly/Matplotlib
│   └── wire_segmentation.py   # Point cloud clustering and segmentation
│
├── README.md
└── requirements.txt
```




---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/hajar6883/catenary-wire-modeling.git
    cd catenary-wire-modeling
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

```bash
python cli.py --input <path-to-lidar-parquet-file>
```

You can customize clustering and fitting parameters inside `presets.py`.

---

## Methodology

1. **Clustering**  
   Using density-based clustering (HDBSCAN or DBSCAN), point clouds are grouped wire-by-wire.

2. **Curve Fitting**  
   Each cluster is fitted with a 3D catenary model. The 2D catenary formula is:
   \[
   y(x) = y_0 + c \cdot \left( \cosh \left(\frac{x - x_0}{c}\right) - 1 \right)
   \]
   Adapted for 3D using parameter optimization.

3. **Visualization**  
   We provide 3D plots of both raw and modeled wires using Plotly and Matplotlib.

---

| **Approach** | Systematic wire segmentation → clustering → catenary fitting |

---

## Future Work

- Handle more complex wire topologies
- Automate parameter tuning
- Improve performance on noisy datasets
- Integrate more advanced 3D fitting techniques

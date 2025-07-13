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

## Proposed Approach

### 1. Clustering

The final pipeline is designed to generalize across all datasets by dynamically selecting the appropriate strategy for segmenting wires based on their spatial arrangement.

#### Flat Wire Configurations (Easy, Hard, Extra Hard)

In these cases, wires are approximately aligned along the same Z level — with minimal vertical stacking. The pipeline uses:

- **Projection to 2D**  
  The 3D point cloud is projected onto the X-Y plane using simple axis selection.

- **PCA Alignment**  
  Principal Component Analysis (PCA) is applied to align the wires horizontally in this 2D space.

- **DBSCAN Clustering**  
  Clustering is then performed along the second PCA axis, which typically separates the parallel wires vertically.  
  This method is both efficient and effective for scenes where wires do not overlap in height.

#### Vertically Stacked / Diagonal Wires (Medium)

In more complex datasets, wires may be stacked vertically or oriented diagonally, where axis-aligned projection is insufficient.

To handle this:

- **HDBSCAN for Layer Detection**  
  A Y-Z projection is used to identify major vertical clusters (upper vs. lower layers) using HDBSCAN.

- **Within Each Layer**:
  - Project the points to X-Y
  - Apply PCA to normalize orientation
  - Run DBSCAN to separate individual wires
  - Assign globally unique wire IDs using a label counter

This two-stage hierarchical approach ensures robust clustering across both simple and complex geometries, avoiding incorrect wire splits or merges — even when wires are diagonally aligned or vertically stacked.



### 2. Curve Fitting

After clustering, each group of wire points is fitted with a 3D catenary curve to model the physical sag of suspended cables.

1. **PCA Projection**  
   Each 3D wire cluster is projected onto its best-fit 2D plane using PCA, simplifying the problem to 2D.

2. **2D Catenary Fit**  
   The catenary equation  
   $$
    y(x) = y_0 + c \cdot \left( \cosh\left(\frac{x - x_0}{c}\right) - 1 \right)
    $$ 
   is fitted using non-linear least squares (`curve_fit`), estimating \( x_0 \), \( y_0 \), and \( c \).

3. **Back-Projection to 3D**  
   The fitted curve is transformed back into 3D using the inverse PCA transform, restoring real-world orientation.

4. **Fit Quality (RMSE)**  
   A KD-tree is used to compute the RMSE between original points and the fitted curve, assessing fit accuracy.

5. **Batch Processing**  
   All wires are processed in a loop. Invalid or poor fits are skipped with optional logging of parameters and RMSE.

This method provides an efficient and physically accurate way to model wire shapes in 3D point clouds.


3. **Visualization**  
   We provide 3D plots of both raw and modeled wires using Plotly and Matplotlib.

---



## Next steps :

While the current approach uses PCA-based projection to simplify curve fitting, an initial attempt was made to fit catenary curves **directly in 3D space** via optimization. This approach proved more complex due to unstable parameter estimation in noisy and unstructured point clouds.

As next steps, future improvements may include:

- **Robust 3D Optimization**  
  Revisit full 3D fitting using advanced solvers with regularization or geometric constraints to improve stability without relying on 2D projection.

- **RANSAC-Inspired Fitting**  
  Incorporate RANSAC-like methods to robustly fit curves in the presence of outliers. By sampling minimal subsets and evaluating fit quality across the full wire segment, such methods can reduce sensitivity to noise and occlusion.

- **Noise Filtering & Denoising**  
  While the example datasets were preprocessed to some extent—making wires visually distinguishable in plots—further filtering could improve curve fitting. Removing residual outliers and smoothing wire segments may enhance model stability and fit accuracy, especially in noisier or raw datasets.

- **Confidence Scoring**  
  Use per-wire RMSE and point density to assign confidence scores to fitted curves, helping downstream applications filter unreliable segments.

These enhancements could help the model better generalize to more challenging, noisy, or incomplete LiDAR datasets.




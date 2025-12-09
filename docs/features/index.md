# Features Overview

SlicerSEEG provides a complete pipeline for automated SEEG electrode localization.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SlicerSEEG Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   POST-OP CT ──────┐                                                        │
│                    ▼                                                        │
│   PRE-OP MRI ─► BRAIN EXTRACTION ─► IMAGE ENHANCEMENT ─► ENSEMBLE VOTING   │
│                (MONAI 3D U-Net)     (7 parallel methods)  (38 variants)     │
│                                                                             │
│                                           ▼                                 │
│                                                                             │
│   EXPORT ◄── TRAJECTORY ◄───── CONFIDENCE ◄── CENTROID DETECTION           │
│   (CSV/3D)   RECONSTRUCTION     ANALYSIS      (Connected Components)        │
│              (Semi-automatic)   (LightGBM)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Brain Extraction

Deep learning-based brain segmentation using MONAI 3D U-Net architecture.

- **Dice Coefficient**: 0.936 ± 0.011
- **Purpose**: Isolate brain tissue, remove skull and non-brain structures
- **Technology**: MONAI framework with PyTorch backend

[Learn more →](brain-extraction.md)

---

### 2. Image Enhancement

Seven parallel enhancement methods ensure robust electrode detection:

| Method | Algorithm | Purpose |
|--------|-----------|---------|
| CTP Enhancement | Frangi vesselness | Tubular structure enhancement |
| Wavelet Denoising | PyWavelets | Edge-preserving noise reduction |
| Adaptive Thresholding | Random Forest | ML-predicted parameters |
| Morphological Ops | Opening/Closing | Shape-based refinement |
| Histogram Enhancement | Statistical | Intensity normalization |
| Percentile Thresholding | Distribution | Robust to outliers |
| Gaussian Filtering | Multi-scale | Smoothing |

[Learn more →](enhancement.md)

---

### 3. Ensemble Consensus

38 segmentation variants are combined through intelligent voting:

- **Global voting** across all enhancement results
- **Multiple confidence levels**: top_mask_1, top_mask_2, consensus_50pct
- **Louvain community detection** for consensus generation

---

### 4. Confidence Analysis

ML-based validation with 38-dimensional feature vectors:

| Feature Category | Examples |
|-----------------|----------|
| **Spatial** | RAS coordinates, hemisphere, relative position |
| **Intensity** | CT mean, std, min, max, percentiles |
| **Topological** | PCA components, KDE density, neighbor distances |
| **Geometric** | Distance to surface, distance to centroid |

[Learn more →](confidence.md)

---

### 5. Trajectory Reconstruction

Enhanced trajectory reconstruction beyond simple interpolation:

- **Semi-Automatic Mode**: Uses detected contacts, fills gaps
- **Manual Mode**: User-defined points, spacing estimation
- **Intelligent Features**: Gap filling, SEEG labeling, spline fitting
- **Visual Output**: Hemisphere-coded colors (Blue/Pink)

[Learn more →](trajectory.md)

---

## Technology Stack

<div class="grid" markdown>

=== "Deep Learning"

    - PyTorch ≥1.10.0
    - MONAI ≥0.9.0
    - LightGBM ≥3.3.0

=== "Image Processing"

    - SimpleITK ≥2.0.0
    - scikit-image ≥0.18.0
    - PyWavelets ≥1.1.0

=== "Scientific"

    - NumPy ≥1.20.0
    - SciPy ≥1.7.0
    - Pandas ≥1.3.0

=== "Visualization"

    - Matplotlib ≥3.4.0
    - Plotly ≥5.0.0
    - NetworkX ≥2.6.0

</div>

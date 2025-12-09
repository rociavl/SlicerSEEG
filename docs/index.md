# SlicerSEEG

**Automated SEEG Electrode Localization for Epilepsy Surgery Planning**

---

## From 4+ Hours to 30 Minutes

SlicerSEEG is a 3D Slicer extension that automates the localization of SEEG (Stereoelectroencephalography) electrodes from post-operative CT scans. Designed for epileptologists and neurosurgeons, this tool dramatically reduces manual processing time while maintaining clinical accuracy standards.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **90% Time Reduction**

    ---

    Process cases in 15-30 minutes instead of 4+ hours of manual work

-   :material-target:{ .lg .middle } **98.8% Accuracy**

    ---

    Sub-millimeter localization within 2mm clinical threshold

-   :material-brain:{ .lg .middle } **Deep Learning**

    ---

    MONAI 3D U-Net for automated brain extraction

-   :material-chart-box:{ .lg .middle } **ML Confidence**

    ---

    38-dimensional feature analysis with LightGBM

</div>

---

## Clinical Impact

| Metric | Performance | Significance |
|--------|-------------|--------------|
| **Localization Accuracy** | 98.8% within 2mm | Gold standard clinical threshold |
| **Detection Sensitivity** | 100% | No electrodes missed |
| **Processing Time** | 15-30 min | 90% reduction vs. manual |
| **Precision** | 97.5% | Minimal false positives |
| **Brain Segmentation** | Dice 0.936 | Excellent structural agreement |

!!! success "Validated at Hospital del Mar"
    Clinical validation performed on 16-patient cohort at Hospital del Mar, Barcelona, Spain.

---

## Processing Pipeline

```
POST-OP CT ──────┐
                 ▼
PRE-OP MRI ─► BRAIN EXTRACTION ─► IMAGE ENHANCEMENT ─► ENSEMBLE VOTING
             (MONAI 3D U-Net)     (7 parallel methods)  (38 variants)
                                          │
                                          ▼
EXPORT ◄── TRAJECTORY ◄───── CONFIDENCE ◄── CENTROID DETECTION
(CSV/3D)   RECONSTRUCTION     ANALYSIS      (Connected Components)
           (Semi-automatic)   (LightGBM)
```

---

## Key Features

### Intelligent Trajectory Reconstruction

The enhanced trajectory reconstruction module goes beyond simple interpolation:

- **Semi-Automatic Mode**: Uses detected contacts, fills gaps automatically
- **Manual Mode**: User defines points, system estimates contacts
- **SEEG Convention**: Proper labeling (deepest = 1)
- **Hemisphere Coloring**: Blue (right) / Pink (left)

[Learn more about Trajectory Reconstruction :material-arrow-right:](features/trajectory.md)

### 38-Variant Ensemble Consensus

Seven parallel enhancement methods generate 38 segmentation variants, combined through intelligent voting for robust detection.

[Explore Image Enhancement :material-arrow-right:](features/enhancement.md)

### ML-Based Confidence Analysis

Each electrode candidate is evaluated across 38 features including spatial, intensity, and topological characteristics.

[Understand Confidence Analysis :material-arrow-right:](features/confidence.md)

---

## Quick Start

=== "Step 1: Install"

    ```python
    # In 3D Slicer Python Console
    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/setup_dependencies.py'
    ).read().decode())
    ```

=== "Step 2: Load Data"

    1. Import post-operative CT scan
    2. (Optional) Load pre-operative MRI
    3. Navigate to **Modules → Segmentation → SEEG ElectrodeLocalization**

=== "Step 3: Process"

    1. Select input volumes
    2. Click **Apply**
    3. Wait 15-30 minutes

=== "Step 4: Review"

    1. Adjust confidence threshold
    2. Reconstruct trajectories (A, B, C...)
    3. Export validated coordinates

[Full Installation Guide :material-arrow-right:](getting-started/installation.md)

---

## Technology Stack

<div class="grid" markdown>

| Deep Learning | Image Processing |
|---------------|------------------|
| PyTorch ≥1.10 | SimpleITK ≥2.0 |
| MONAI ≥0.9 | scikit-image ≥0.18 |
| LightGBM ≥3.3 | PyWavelets ≥1.1 |

| Scientific | Visualization |
|------------|---------------|
| NumPy ≥1.20 | Matplotlib ≥3.4 |
| SciPy ≥1.7 | Plotly ≥5.0 |
| Pandas ≥1.3 | NetworkX ≥2.6 |

</div>

---

## Citation

If you use SlicerSEEG in your research, please cite:

```bibtex
@mastersthesis{avalos2025seeg,
  title     = {Medical Software Module in 3D Slicer for Automatic
               Segmentation and Trajectory Reconstruction of SEEG
               Electrodes Using AI and Data Science},
  author    = {Ávalos Morillas, Rocío},
  year      = {2025},
  school    = {Universitat Politècnica de Catalunya},
  type      = {Bachelor's Thesis},
  address   = {Barcelona, Spain},
  url       = {https://github.com/rociavl/SlicerSEEG}
}
```

[Full Citation Information :material-arrow-right:](about/citation.md)

---

<div align="center">

**SlicerSEEG** — Transforming epilepsy surgery planning through AI

*Made with science in Barcelona*

[:fontawesome-brands-github: GitHub](https://github.com/rociavl/SlicerSEEG){ .md-button }
[:material-email: Contact](mailto:rocio.avalos029@gmail.com){ .md-button }

</div>

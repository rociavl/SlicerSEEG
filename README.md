# SlicerSEEG

**Automated SEEG Electrode Localization for Epilepsy Surgery Planning**

[![3D Slicer](https://img.shields.io/badge/3D%20Slicer-5.0+-blue.svg)](https://slicer.org/)
[![License](https://img.shields.io/badge/License-Slicer-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Clinical Validation](#clinical-validation)
- [Research & Citation](#research--citation)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## Overview

SlicerSEEG is a 3D Slicer extension that automates the localization of SEEG (Stereoelectroencephalography) electrodes from post-operative CT scans. Designed for epileptologists and neurosurgeons, this tool dramatically reduces manual processing time from **over 4 hours to approximately 30 minutes** while maintaining clinical accuracy standards.

### Clinical Impact

| Metric | Performance |
|--------|-------------|
| **Localization Accuracy** | 98.8% within 2mm clinical threshold |
| **Processing Time** | 15-30 minutes per case |
| **Sensitivity** | 100% electrode detection |
| **Time Savings** | ~90% reduction vs. manual processing |


## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Automated Brain Extraction** | Deep learning-based segmentation using MONAI 3D U-Net |
| **Electrode Enhancement** | 7 parallel image processing approaches with adaptive thresholding |
| **Ensemble Consensus** | Global voting from 38 segmentation variants for robust detection |
| **Confidence Analysis** | ML-based electrode validation with graduated certainty scores |
| **Interactive Viewer** | Real-time visualization with adjustable confidence thresholds |
| **Trajectory Reconstruction** | Automated pathway mapping with multi-algorithm consensus |

### Processing Pipeline

```
Post-Op CT Scan → Brain Extraction → Electrode Enhancement →
Global Voting (38 variants) → Confidence Analysis →
Interactive Review → Trajectory Reconstruction → Export Results
```

---

## Installation

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **3D Slicer** | Version 5.0 or later ([Download](https://download.slicer.org/)) |
| **RAM** | 8GB minimum (16GB recommended) |
| **Storage** | 2GB free space |
| **GPU** | Optional (CUDA-compatible GPU for faster processing) |

### Installation Steps

#### Step 1: Install Python Dependencies

SlicerSEEG requires additional Python packages. Open 3D Slicer's Python Interactor (**View → Python Interactor**) and run:

```python
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/setup_dependencies.py').read().decode())
```

⏱️ **Installation time**: 2-5 minutes. Restart 3D Slicer when complete.

<details>
<summary><b>Alternative: Manual Package Installation</b></summary>

```python
import subprocess, sys

packages = [
    'lightgbm',  # Confidence analysis
    'torch',     # Brain segmentation
    'monai',     # Medical imaging AI
    'networkx',  # Trajectory analysis
    'plotly'     # Interactive visualizations
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
```

See [requirements.txt](requirements.txt) for the complete dependency list.
</details>

#### Step 2: Install Extension

**Option A: Extension Manager (Coming Soon)**
1. Open **View → Extension Manager**
2. Search for "SlicerSEEG"
3. Click **Install** and restart Slicer

**Option B: Manual Installation**
1. Download the [latest release](https://github.com/rociavl/SlicerSEEG/releases)
2. Extract the archive
3. In Slicer: **Edit → Application Settings → Modules**
4. Click **Add** (>>) under "Additional module paths"
5. Select the `SEEG_ElectrodeLocalization` folder
6. Click **OK** and restart Slicer

#### Step 3: Verify Installation

After restart, navigate to **Modules → Segmentation → SEEG ElectrodeLocalization**.

✅ If the module appears, installation was successful!

---

## Quick Start

Get started with SlicerSEEG in 4 simple steps:

### 1. Load Data
- Import post-operative CT scan with SEEG electrodes
- (Optional) Load pre-operative MRI for brain mask generation

### 2. Configure Module
- Navigate to **Modules → Segmentation → SEEG ElectrodeLocalization**
- Select input volumes:
  - **MRI Input**: For brain mask generation (or use existing mask)
  - **CT Input**: Post-operative scan with electrodes
- (Optional) Set custom output folder name

### 3. Run Processing
- Click **Apply**
- Monitor progress in Python console
- ⏱️ Processing time: ~15-30 minutes depending on scan quality

### 4. Review Results
- **Confidence Viewer**: Adjust threshold slider to filter electrode candidates
- **3D View**: Visualize detected electrodes in Slicer's 3D viewer
- **Export**: Save validated coordinates for surgical planning

---

## Usage

### Understanding the Output

Results are automatically saved to `~/Documents/SEEG_Results/[folder_name]/`:

```
SEEG_Results/
├── 📁 Brain_mask/              # Automated brain segmentation
├── 📁 Enhanced_masks/          # 38 processed segmentation variants
├── 📁 Global_masks/            # Top consensus masks
│   ├── top_mask_1_*.nrrd      # Highest confidence mask
│   ├── top_mask_2_*.nrrd      # Second-best mask
│   └── consensus_50pct_*.nrrd # 50% voting threshold
└── 📁 Confidence_Analysis/     # Electrode validation results
    ├── target_features_*.csv          # Feature vectors (38 dimensions)
    ├── confidence_predictions_*.csv   # ML confidence scores
    └── confidence_summary_*.txt       # Summary statistics
```

### Advanced Features

<details>
<summary><b>🔍 Trajectory Analysis</b></summary>

Reconstruct complete electrode trajectories from detected points:

1. Load electrode markups in 3D Slicer
2. Navigate to the **Trajectory Analysis** section
3. Select markup node with electrode points
4. (Optional) Specify trajectory IDs to analyze specific electrodes
5. Click **Generate Reports and CSV** for detailed analysis
6. Click **Create Trajectory Lines** for 3D visualization

**Output**: Trajectory metrics, electrode spacing, and 3D pathway models

</details>

<details>
<summary><b>⚙️ Confidence Threshold Tuning</b></summary>

Optimize electrode detection sensitivity for your data:

- **Slider Control**: Adjust confidence threshold (default: 0.05)
- **Higher Thresholds**: Fewer false positives, stricter validation
- **Lower Thresholds**: More candidates, broader coverage
- **Real-Time Stats**: View detection metrics in the Confidence Viewer panel

**Recommendation**: Start with default (0.05) and adjust based on visual inspection

</details>

### Clinical Workflow Integration

**Input Formats**: DICOM, NRRD, NIFTI
**Output Formats**: CSV coordinates, 3D Slicer markups, NRRD masks
**Compatibility**: Direct integration with 3D Slicer markup tools and surgical planning modules

---

## Technical Details

### Processing Pipeline Architecture

| Step | Method | Purpose |
|------|--------|---------|
| **1. Brain Extraction** | MONAI 3D U-Net | Deep learning segmentation to isolate brain tissue |
| **2. Image Enhancement** | 7 parallel approaches | Adaptive thresholding for electrode visibility |
| **3. Threshold Prediction** | Random Forest regression | Optimal parameter selection per scan |
| **4. Global Voting** | Ensemble consensus | Aggregate 38 segmentation variants |
| **5. Confidence Analysis** | LightGBM classifier | 38-dimensional feature validation |
| **6. Trajectory Reconstruction** | DBSCAN + Louvain | Community detection for electrode pathways |

---

## Clinical Validation

**Study Details**: 8-patient cohort at Hospital del Mar (Barcelona, Spain)

| Metric | Result |
|--------|--------|
| **Localization Accuracy** | 98.8% within 2mm clinical threshold |
| **Sensitivity** | 100% electrode detection rate |
| **Processing Time** | 15-30 minutes per case |
| **False Positive Rate** | <5% with confidence filtering |
| **Brain Segmentation** | 0.936 ± 0.011 Dice coefficient |
| **Manual Time Saved** | ~90% reduction (4+ hours → 30 minutes) |

### Technology Stack

<details>
<summary><b>View Complete Dependencies</b></summary>

**Core Libraries**
- NumPy ≥1.20.0, SciPy ≥1.7.0, Pandas ≥1.3.0
- SimpleITK ≥2.0.0, scikit-image ≥0.18.0, scikit-learn ≥1.0.0

**Machine Learning**
- LightGBM ≥3.3.0 (Confidence analysis)
- PyTorch ≥1.10.0 (Brain segmentation)
- MONAI ≥0.9.0 (Medical imaging AI)

**Visualization & Analysis**
- NetworkX ≥2.6.0 (Trajectory graphs)
- Matplotlib ≥3.4.0 (Plotting)
- Plotly ≥5.0.0 (Interactive visualizations)

*Note: VTK and Qt are provided by 3D Slicer*

See [requirements.txt](requirements.txt) for complete list.

</details>

---

## Research & Citation

### Published Work

This extension is based on research conducted at Universitat Politècnica de Catalunya:

> **Ávalos, R. (2025).** "Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science."
> *Bachelor's Thesis, Universitat Politècnica de Catalunya*.

### Contributing Institutions

| Institution | Role |
|-------------|------|
| **Hospital del Mar** (Barcelona) | Clinical validation and deployment |
| **Center for Brain and Cognition, UPF** | Research collaboration |
| **Universitat Politècnica de Catalunya** | Technical development |

### How to Cite

If you use SlicerSEEG in your research, please cite:

```bibtex
@mastersthesis{avalos2025seeg,
  title={Medical Software Module in 3D Slicer for Automatic Segmentation and
         Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science},
  author={Ávalos Morillas, Rocío},
  year={2025},
  school={Universitat Politècnica de Catalunya},
  type={Bachelor's Thesis},
  url={https://github.com/rociavl/SlicerSEEG}
}
```

---

## Contributing

Contributions are welcome from the medical imaging and epilepsy research communities!

### Development Setup

```bash
# Clone repository
git clone https://github.com/rociavl/SlicerSEEG.git
cd SlicerSEEG

# Install development dependencies
pip install -r requirements.txt

# Add to Slicer as described in Installation section
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/YourFeature`
3. **Commit** your changes: `git commit -m 'Add YourFeature'`
4. **Push** to the branch: `git push origin feature/YourFeature`
5. **Open** a Pull Request with a clear description

### Development Priorities

We're particularly interested in contributions for:

- ✅ Multi-center validation across electrode manufacturers
- ✅ Integration with fMRI and PET modalities
- ✅ Real-time processing optimization
- ✅ Extended trajectory analysis algorithms
- ✅ Improved visualization tools

---

## Support

### Get Help

| Resource | Description |
|----------|-------------|
| **Bug Reports** | [GitHub Issues](https://github.com/rociavl/SlicerSEEG/issues) |
| **Feature Requests** | [GitHub Discussions](https://github.com/rociavl/SlicerSEEG/discussions) |
| **Clinical Questions** | Contact Hospital del Mar Epilepsy Unit |

### Contact

**Rocío Ávalos Morillas**
*Biomedical Engineer, Universitat Politècnica de Catalunya*

- 📧 Email: [rocio.avalos029@gmail.com](mailto:rocio.avalos029@gmail.com)
- 💼 LinkedIn: [Rocío Ávalos Morillas](https://www.linkedin.com/in/rocío-ávalos-morillas-04a5372b1/)
- 💻 GitHub: [@rociavl](https://github.com/rociavl)

---

## License

This project is licensed under the same terms as 3D Slicer. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was made possible by the collaboration of exceptional individuals and institutions:

| Contributor | Role | Affiliation |
|-------------|------|-------------|
| **Dr. Alessandro Principe** | Clinical guidance and validation | Hospital del Mar |
| **Justo Montoya-Gálvez** | Computational neuroscience collaboration | UPF Center for Brain and Cognition |
| **Prof. Christian Mata** | Academic supervision | Universitat Politècnica de Catalunya |
| **3D Slicer Community** | Open-source platform and development support | — |

---

## Screenshots

### Confidence-Based Electrode Visualization
![Confidence Viewer](https://github.com/user-attachments/assets/be33d580-feb4-4caa-9a48-30ebd59ee7e1)

*Interactive confidence threshold adjustment with real-time filtering and statistical analysis. Adjust the slider to optimize detection sensitivity for your clinical needs.*

---

### Automated Processing Pipeline
![Processing Results](https://github.com/user-attachments/assets/4e0f3fa7-2de5-4efc-b5d4-10d8878caf77)

*End-to-end automated workflow from CT scan input to validated electrode coordinates with 3D visualization and confidence scoring.*

---

<div align="center">

**Questions or need help?** Open an issue on [GitHub](https://github.com/rociavl/SlicerSEEG/issues) or contact the development team.

**Made with ❤️ for the epilepsy research community**

</div>
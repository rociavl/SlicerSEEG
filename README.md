# SlicerSEEG

**Automated SEEG Electrode Localization for Epilepsy Surgery Planning**

[![3D Slicer](https://img.shields.io/badge/3D%20Slicer-5.0+-blue.svg)](https://slicer.org/)
[![License](https://img.shields.io/badge/License-Slicer-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

---

## Overview

SlicerSEEG is a 3D Slicer extension that automates the localization of SEEG (Stereoelectroencephalography) electrodes from post-operative CT scans. The extension reduces manual processing time from over 4 hours to approximately 30 minutes while maintaining clinical accuracy standards (98.8% within 2mm threshold).

**Clinical Validation**: Active deployment at Hospital del Mar Epilepsy Unit, Barcelona, Spain.

### Key Capabilities

- **Automated Brain Extraction**: Deep learning-based segmentation using MONAI 3D U-Net
- **Electrode Enhancement**: 7 parallel image processing approaches with adaptive thresholding
- **Ensemble Consensus**: Global voting from 38 segmentation variants
- **Confidence Analysis**: Machine learning-based electrode validation with graduated certainty scores
- **Interactive Viewer**: Real-time electrode visualization with adjustable confidence thresholds
- **Trajectory Reconstruction**: Automated pathway mapping with multi-algorithm consensus

---

## Installation

### Requirements

- **3D Slicer**: Version 5.0 or later ([Download](https://download.slicer.org/))
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CUDA-compatible GPU improves performance)

### Step 1: Install Python Dependencies

SlicerSEEG requires additional Python packages. Open 3D Slicer's Python Interactor (**View → Python Interactor**) and execute:

```python
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/setup_dependencies.py').read().decode())
```

**Installation time**: 2-5 minutes. Restart 3D Slicer when complete.

<details>
<summary><b>Manual Installation (Click to expand)</b></summary>

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

See [requirements.txt](requirements.txt) for complete dependency list.
</details>

### Step 2: Install Extension

#### Option A: Extension Manager (Coming Soon)
1. Open **View → Extension Manager**
2. Search for "SlicerSEEG"
3. Click **Install** and restart Slicer

#### Option B: Manual Installation
1. Download the [latest release](https://github.com/rociavl/SlicerSEEG/releases)
2. Extract the archive
3. In Slicer: **Edit → Application Settings → Modules**
4. Click **Add** (>>) under "Additional module paths"
5. Select the `SEEG_ElectrodeLocalization` folder
6. Click **OK** and restart Slicer

### Step 3: Verify Installation

After restart, navigate to **Modules → Segmentation → SEEG ElectrodeLocalization**. If the module appears, installation was successful.

---

## Usage

### Quick Start

1. **Load Data**
   - Import post-operative CT scan with SEEG electrodes
   - (Optional) Load pre-operative MRI for brain mask

2. **Configure Module**
   - Navigate to **Modules → Segmentation → SEEG ElectrodeLocalization**
   - Select input volumes:
     - **MRI Input**: For brain mask generation (or existing mask)
     - **CT Input**: Post-operative scan with electrodes
   - (Optional) Set custom output folder name

3. **Run Processing**
   - Click **Apply**
   - Monitor progress in Python console (~30 minutes)

4. **Review Results**
   - **Confidence Viewer**: Adjust threshold slider to filter electrode candidates
   - **3D View**: Visualize detected electrodes in Slicer's 3D viewer
   - **Export**: Save validated coordinates for surgical planning

### Output Structure

Results are saved to `~/Documents/SEEG_Results/[folder_name]/`:

```
SEEG_Results/
├── Brain_mask/              # Automated brain segmentation
├── Enhanced_masks/          # 38 processed variants
├── Global_masks/            # Top consensus masks
│   ├── top_mask_1_*.nrrd
│   ├── top_mask_2_*.nrrd
│   └── consensus_50pct_*.nrrd
└── Confidence_Analysis/     # Electrode validation
    ├── target_features_*.csv
    ├── confidence_predictions_*.csv
    └── confidence_summary_*.txt
```

### Advanced Features

<details>
<summary><b>Trajectory Analysis</b></summary>

Reconstruct complete electrode trajectories:

1. Load electrode markups in 3D Slicer
2. In the **Trajectory Analysis** section:
   - Select markup node with electrode points
   - (Optional) Specify trajectory IDs to analyze
3. Click **Generate Reports and CSV** for analysis
4. Click **Create Trajectory Lines** for 3D visualization

</details>

<details>
<summary><b>Confidence Threshold Tuning</b></summary>

Optimize electrode detection for your data:

- Use slider to adjust confidence threshold (default: 0.05)
- Higher thresholds: Fewer false positives, stricter validation
- Lower thresholds: More candidates, broader coverage
- View real-time statistics in the Confidence Viewer panel

</details>

---

## Technical Details

### Processing Pipeline

1. **Brain Extraction** - MONAI 3D U-Net deep learning segmentation
2. **Image Enhancement** - 7 parallel processing approaches with adaptive thresholding
3. **Threshold Prediction** - Random Forest regression for optimal parameters
4. **Global Voting** - Ensemble consensus from 38 segmentation variants
5. **Confidence Analysis** - LightGBM classification with 38-dimensional features
6. **Trajectory Reconstruction** - DBSCAN + Louvain community detection

### Performance Metrics

**Clinical Validation (8-patient cohort, Hospital del Mar)**
- Localization Accuracy: 98.8% within 2mm clinical threshold
- Sensitivity: 100% electrode detection
- Processing Time: 15-30 minutes per case
- False Positive Rate: <5% with confidence filtering
- Brain Segmentation: 0.936 ± 0.011 Dice coefficient

### Dependencies

**Core Libraries**
- NumPy ≥1.20.0, SciPy ≥1.7.0, Pandas ≥1.3.0
- SimpleITK ≥2.0.0, scikit-image ≥0.18.0, scikit-learn ≥1.0.0

**Machine Learning**
- LightGBM ≥3.3.0, PyTorch ≥1.10.0, MONAI ≥0.9.0

**Visualization & Analysis**
- NetworkX ≥2.6.0, Matplotlib ≥3.4.0, Plotly ≥5.0.0

*Note: VTK and Qt are provided by 3D Slicer*

---

## Clinical Workflow

### For Epileptologists & Neurosurgeons

**Input**: Post-operative CT scan (DICOM or NRRD format)

**Processing**: 
- Automatic brain extraction and electrode detection
- Confidence-based validation with interactive review
- Export of validated coordinates in standard formats

**Output**: 
- Electrode coordinates with confidence scores
- 3D visualizations for surgical planning
- Statistical reports and quality metrics

**Integration**: Direct compatibility with 3D Slicer's markup tools and surgical planning modules

---

## Research & Development

### Published Work

This extension is based on:

> Ávalos, R. (2025). "Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science." *Bachelor's Thesis, Universitat Politècnica de Catalunya*.

### Contributing Institutions

- **Hospital del Mar** (Barcelona, Spain) - Clinical validation and deployment
- **Center for Brain and Cognition, UPF** - Research collaboration  
- **Universitat Politècnica de Catalunya** - Technical development

### Citation

```bibtex
@mastersthesis{avalos2025seeg,
  title={Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science},
  author={Ávalos Morillas, Rocío},
  year={2025},
  school={Universitat Politècnica de Catalunya},
  type={Bachelor's Thesis},
  url={https://github.com/rociavl/SlicerSEEG}
}
```

---

## Contributing

Contributions are welcome from the medical imaging and epilepsy research communities.

### Development Setup

```bash
# Clone repository
git clone https://github.com/rociavl/SlicerSEEG.git
cd SlicerSEEG

# Install development dependencies
pip install -r requirements.txt

# Add to Slicer as described in Installation
```

### Contribution Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

### Development Priorities

- Multi-center validation across electrode manufacturers
- Integration with fMRI and PET modalities
- Real-time processing optimization
- Extended trajectory analysis algorithms

---

## Support

### Documentation

- **User Guide**: Detailed usage instructions and clinical workflows
- **API Reference**: Developer documentation for extension integration
- **Troubleshooting**: Common issues and solutions

### Contact

**Rocío Ávalos Morillas**  
*Biomedical Engineer, Universitat Politècnica de Catalunya*

- 📧 Email: [rocio.avalos029@gmail.com](mailto:rocio.avalos029@gmail.com)
- 🔗 LinkedIn: [Rocío Ávalos](https://www.linkedin.com/in/rocío-ávalos-morillas-04a5372b1/)
- 🐙 GitHub: [@rociavl](https://github.com/rociavl)

### Issue Tracking

- **Bug Reports**: [GitHub Issues](https://github.com/rociavl/SlicerSEEG/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/rociavl/SlicerSEEG/discussions)
- **Clinical Questions**: Contact Hospital del Mar Epilepsy Unit

---

## License

This project is licensed under the same terms as 3D Slicer. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Dr. Alessandro Principe** - Clinical guidance and validation (Hospital del Mar)
- **Justo Montoya-Gálvez** - Computational neuroscience collaboration (UPF CBC)
- **Prof. Christian Mata** - Academic supervision (UPC)
- **3D Slicer Community** - Open-source platform and development support

---

## Extension Screenshots

### Confidence-Based Electrode Visualization
![Confidence Viewer](https://github.com/user-attachments/assets/be33d580-feb4-4caa-9a48-30ebd59ee7e1)
*Interactive confidence threshold adjustment with real-time filtering and statistical analysis*

### Automated Processing Pipeline
![Processing Results](https://github.com/user-attachments/assets/4e0f3fa7-2de5-4efc-b5d4-10d8878caf77)
*End-to-end automated workflow from CT scan to validated electrode coordinates*

---

**For technical support or clinical implementation questions, please open an issue on GitHub or contact the development team directly.**
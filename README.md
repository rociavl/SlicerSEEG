# SlicerSEEG - Automated SEEG Electrode Localization

[![3D Slicer](https://img.shields.io/badge/3D%20Slicer-5.0+-blue.svg)](https://slicer.org/)
[![License](https://img.shields.io/badge/License-Slicer-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

## Overview

SlicerSEEG is an automated SEEG (Stereoelectroencephalography) electrode localization system for epilepsy surgery planning. This 3D Slicer extension transforms the manual 4+ hour electrode identification process into a 30-minute automated workflow while maintaining clinical accuracy standards.

## 🎯 Clinical Impact

- **Time Reduction**: 4+ hours → 30 minutes (87% reduction in specialist workload)
- **Accuracy**: 98.8% localization accuracy within 2mm clinical threshold
- **Patient Safety**: Reduced electrode implantation duration and infection risk
- **Clinical Deployment**: Active use at Hospital del Mar Epilepsy Unit, Barcelona

## ✨ Key Features

### 🧠 **Automated Brain Extraction**
- Deep learning-based brain segmentation from post-operative CT scans
- MONAI-based 3D U-Net architecture
- Handles metal artifacts and electrode interference

### 🖼️ **Multi-Modal Image Enhancement**
- 7 specialized image processing approaches
- Adaptive thresholding with Random Forest prediction
- Optimized for electrode visibility in challenging CT conditions

### 🗳️ **Global Voting Ensemble**
- Consensus formation from 38 segmentation variants
- Intelligent mask selection and quality assessment
- Redundant detection coverage for robust performance

### 🎯 **Confidence-Based Authentication**
- Machine learning classification with graduated certainty scores
- 38-dimensional feature space for electrode validation
- Conservative design preserving clinical decision-making authority

### 📊 **Interactive Confidence Viewer**
- Real-time electrode visualization with adjustable confidence thresholds
- Statistical analysis of electrode candidates
- Seamless integration with 3D Slicer markup tools

### 🛤️ **Complete Trajectory Reconstruction**
- End-to-end electrode pathway mapping from cortical entry to deep brain targets
- Multi-algorithm consensus framework (DBSCAN + Louvain)
- Automated quality assessment and clinical validation

## 🚀 Installation

### Option 1: Extension Manager (Recommended)
1. Open 3D Slicer (version 5.0 or later)
2. Go to **View** → **Extension Manager**
3. Search for **"SlicerSEEG"**
4. Click **Install** and restart Slicer

### Option 2: Manual Installation
```bash
git clone https://github.com/rociavl/SlicerSEEG.git
# Follow 3D Slicer extension development guidelines for building
```

## 📖 Quick Start Guide

### 1. **Load Your Data**
- Import post-operative CT scan containing SEEG electrodes
- Optionally load pre-operative MRI for brain mask (automatic extraction available)

### 2. **Launch Module**
- Navigate to **Modules** → **Segmentation** → **SEEG masking**
- Select input volumes (CT + optional brain mask)

### 3. **Configure Processing**
- Set output folder name (optional)
- Keep default settings for standard clinical use

### 4. **Run Automated Analysis**
- Click **Apply** to start the 6-stage processing pipeline
- Monitor progress through console output (~30 minutes total)

### 5. **Review Results**
- **Brain Mask**: Automated brain extraction results
- **Enhanced Masks**: 38 processed segmentation variants
- **Global Masks**: Top-performing consensus masks
- **Confidence Analysis**: Interactive electrode candidate viewer

### 6. **Interactive Validation**
- Use the **Confidence Viewer** to:
  - Adjust confidence threshold with slider
  - View electrode statistics in real-time
  - Export validated coordinates for surgical planning

## 📁 Output Structure

```
SEEG_Results/
├── Brain_mask/
│   └── BrainMask_[CT_Name].nrrd
├── Enhanced_masks/
│   ├── [ProcessingMethod1]_[CT_Name].nrrd
│   ├── [ProcessingMethod2]_[CT_Name].nrrd
│   └── ... (38 total variants)
├── Global_masks/
│   ├── top_mask_1_[CT_Name].nrrd
│   ├── top_mask_2_[CT_Name].nrrd
│   └── consensus_50pct_[CT_Name].nrrd
└── Confidence_Analysis/
    ├── target_features_[CT_Name]_top_mask_1.csv
    ├── confidence_predictions_[CT_Name].csv
    └── confidence_summary_[CT_Name].txt
```

## 🔧 Technical Details

### System Requirements
- **3D Slicer**: Version 5.0 or later
- **RAM**: 8GB+ recommended (16GB for large datasets)
- **Storage**: 2GB free space for processing cache
- **GPU**: Optional (improves brain segmentation performance)

### Pipeline Architecture
1. **Brain Extraction** (MONAI 3D U-Net)
2. **Image Enhancement** (7 parallel approaches)
3. **Threshold Prediction** (Random Forest regression)
4. **Global Voting** (Ensemble consensus)
5. **Contact Authentication** (LightGBM classification)
6. **Trajectory Reconstruction** (Multi-algorithm consensus)

### Key Dependencies
- **Core**: NumPy, SciPy, scikit-learn, pandas
- **Medical Imaging**: SimpleITK, VTK, MONAI
- **Machine Learning**: LightGBM, PyTorch
- **3D Slicer**: Built-in Python environment

## 🏥 Clinical Workflow Integration

### For Neurologists
- **Input**: Post-operative CT scan
- **Processing**: Automated 30-minute analysis
- **Output**: Validated electrode coordinates with confidence scores
- **Integration**: Direct export to surgical planning systems

### For Neurosurgeons  
- **Review**: Interactive 3D visualization in Slicer
- **Validation**: Confidence-based electrode acceptance/rejection
- **Planning**: Complete trajectory information for surgical approach

### For Medical Imaging Specialists
- **Quality Control**: Automated processing reports and statistics
- **Customization**: Adjustable confidence thresholds and processing parameters
- **Documentation**: Comprehensive analysis logs and validation metrics

## 📊 Performance Metrics

### Clinical Validation (8-patient cohort)
- **Localization Accuracy**: 98.8% within 2mm clinical threshold
- **Sensitivity**: 100% electrode detection on held-out patients
- **Processing Time**: 15-30 minutes depending on electrode count
- **False Positive Rate**: <5% with confidence-based filtering

### Algorithm Performance
- **Brain Segmentation**: 0.936 ± 0.011 Dice coefficient
- **Confidence Calibration**: Conservative design with graduated clinical utility
- **Trajectory Reconstruction**: 75% automated success rate

## 🔬 Research & Development

### Published Research
This extension is based on the bachelor's thesis:

> Ávalos, R. (2025). "Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science." Bachelor's Thesis, Universitat Politècnica de Catalunya.

### Contributing Institutions
- **Hospital del Mar** - Clinical validation and deployment
- **UPF Center for Brain and Cognition** - Research collaboration
- **Universitat Politècnica de Catalunya** - Technical development

## 🤝 Contributing

We welcome contributions from the medical imaging and epilepsy research communities:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Areas
- Multi-center validation across different electrode manufacturers
- Integration with additional imaging modalities (fMRI, PET)
- Real-time processing optimization
- Extended trajectory analysis algorithms

## 📄 License

This project is licensed under the same terms as 3D Slicer. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hospital del Mar Epilepsy Unit** - Clinical expertise and validation data
- **Dr. Alessandro Principe** - Clinical guidance and neurosurgical insights  
- **Justo Montoya-Gálvez** - Computational neuroscience collaboration
- **Prof. Christian Mata** - Academic supervision and project guidance
- **3D Slicer Community** - Open-source platform and development support

## 📞 Contact

**Rocío Ávalos Morillas**  
*Biomedical Engineer*  
*Universitat Politècnica de Catalunya*

- 📧 Email: rocio.avalos029@gmail.com
- 🔗 LinkedIn: [Rocío Ávalos](https://www.linkedin.com/in/roc%C3%ADo-%C3%A1valos-morillas-04a5372b1/)
- 🐙 GitHub: [@rociavl](https://github.com/rociavl)

## 📋 Citation

If you use this work in your research, please cite:

```bibtex
@misc{avalos2025seeg,
  title={Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science},
  author={Ávalos Morillas, Rocío},
  year={2025},
  institution={Universitat Politècnica de Catalunya},
  url={https://github.com/rociavl/SlicerSEEG}
}
```

## 📊 Extension Demonstration

### Confidence-Based Electrode Visualization
![SEEG Confidence Viewer](https://github.com/user-attachments/assets/be33d580-feb4-4caa-9a48-30ebd59ee7e1)
*Interactive confidence threshold adjustment with real-time electrode filtering and statistical analysis*

### Complete Pipeline Results
![SEEG Processing Pipeline](https://github.com/user-attachments/assets/4e0f3fa7-2de5-4efc-b5d4-10d8878caf77)
*End-to-end automated processing from CT scan to validated electrode coordinates with clinical-grade accuracy*

## 📞 Support & Contact

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/rociavl/SlicerSEEG/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/rociavl/SlicerSEEG/discussions)
- **📧 Clinical Inquiries**: Contact Hospital del Mar Epilepsy Unit
- **🎓 Academic Collaboration**: Contact UPF Center for Brain and Cognition

## 📚 Documentation

- **User Guide**: [docs/UserGuide.md](docs/UserGuide.md)
- **Developer Guide**: [docs/DeveloperGuide.md](docs/DeveloperGuide.md)
- **API Reference**: [docs/API.md](docs/API.md)
- **Clinical Protocols**: [docs/ClinicalWorkflow.md](docs/ClinicalWorkflow.md)

## 🔗 Related Resources

- [3D Slicer Homepage](https://slicer.org/)
- [SEEG Methodology Overview](https://doi.org/example-reference)
- [Epilepsy Surgery Planning Guidelines](https://doi.org/example-clinical-reference)
- [Medical Image Analysis Best Practices](https://doi.org/example-technical-reference)

---

**⚡ Transform your SEEG electrode localization workflow today with SlicerSEEG!**

*For technical support or clinical implementation questions, please refer to our documentation or contact the development team through GitHub.*
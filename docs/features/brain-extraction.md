# Brain Extraction

Automated brain segmentation using deep learning.

---

## Overview

Brain extraction is the first step in the SlicerSEEG pipeline. It isolates brain tissue from the skull and other non-brain structures, which is essential for accurate electrode localization.

---

## Technology

### MONAI 3D U-Net

SlicerSEEG uses the [MONAI](https://monai.io/) framework for brain segmentation:

- **Architecture**: 3D U-Net with residual connections
- **Backend**: PyTorch
- **Training Data**: Medical imaging datasets

### Performance

| Metric | Value |
|--------|-------|
| **Dice Coefficient** | 0.936 ± 0.011 |
| **Processing Time** | ~5 minutes |
| **GPU Acceleration** | Supported (CUDA) |

---

## How It Works

### Input

- **Pre-operative MRI** (preferred) - Better soft tissue contrast
- **Post-operative CT** (alternative) - If MRI unavailable

### Process

1. **Preprocessing**: Intensity normalization, resampling
2. **Inference**: 3D U-Net predicts brain probability map
3. **Post-processing**: Thresholding, morphological cleanup
4. **Output**: Binary brain mask

### Output

```
Brain_mask/
└── brain_mask_[timestamp].nrrd
```

---

## Usage

### Automatic (Recommended)

When you click **Apply** in the main module, brain extraction runs automatically if:

- An MRI volume is selected
- No existing brain mask is provided

### Using Existing Mask

If you have a pre-computed brain mask:

1. Load the mask volume into Slicer
2. Select it in the **MRI Input** dropdown
3. The system will use it directly (skips extraction)

---

## Technical Details

### Preprocessing Steps

```python
# Intensity normalization
normalized = (image - mean) / std

# Resampling to isotropic resolution
resampled = resample_to_spacing(image, [1.0, 1.0, 1.0])
```

### Post-processing

1. **Thresholding**: Probability > 0.5 → brain
2. **Largest component**: Keep only largest connected region
3. **Hole filling**: Fill internal cavities
4. **Morphological closing**: Smooth boundaries

---

## Troubleshooting

### Poor Segmentation Quality

!!! warning "Common Issues"
    - **Motion artifacts**: Rescan if possible
    - **Low resolution**: Use higher resolution input
    - **Wrong modality**: MRI works better than CT

### Memory Issues

For large volumes:

```python
# Reduce resolution temporarily
import SimpleITK as sitk
resampled = sitk.Resample(image, [128, 128, 128])
```

### GPU Not Detected

Ensure CUDA is properly installed:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## References

- [MONAI Documentation](https://docs.monai.io/)
- [3D U-Net Architecture](https://arxiv.org/abs/1606.06650)

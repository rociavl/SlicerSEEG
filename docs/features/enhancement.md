# Image Enhancement

Seven parallel methods for robust electrode detection.

---

## Overview

SEEG electrodes appear as high-intensity linear structures in CT scans. However, detecting them reliably requires multiple enhancement approaches to handle varying image quality and artifacts.

SlicerSEEG applies **7 parallel enhancement methods**, generating **38 segmentation variants** that are combined through ensemble voting.

---

## Enhancement Methods

### 1. CTP Enhancement (Frangi Filter)

Enhances tubular structures using the Frangi vesselness filter.

```python
# Frangi vesselness filter parameters
frangi_enhanced = frangi(
    image,
    sigmas=range(1, 5),
    alpha=0.5,
    beta=0.5,
    gamma=15
)
```

**Best for**: Linear electrode shafts

---

### 2. Wavelet Denoising

Multi-scale denoising using PyWavelets.

```python
import pywt

# Wavelet decomposition
coeffs = pywt.wavedec3(image, 'db4', level=3)

# Threshold detail coefficients
# Reconstruct denoised image
```

**Best for**: Noisy CT scans

---

### 3. Adaptive Thresholding

ML-predicted optimal threshold using Random Forest regression.

| Input Feature | Purpose |
|--------------|---------|
| Image histogram | Intensity distribution |
| Mean/std intensity | Global statistics |
| Edge density | Structure complexity |

**Best for**: Variable image quality

---

### 4. Morphological Operations

Shape-based refinement using mathematical morphology.

```python
from scipy.ndimage import binary_opening, binary_closing

# Opening removes small bright spots
opened = binary_opening(mask, structure=sphere(2))

# Closing fills small holes
closed = binary_closing(opened, structure=sphere(2))
```

**Best for**: Cleaning segmentation artifacts

---

### 5. Histogram Enhancement

Statistical intensity normalization.

```python
# Percentile-based windowing
p1, p99 = np.percentile(image, [1, 99])
enhanced = np.clip(image, p1, p99)
enhanced = (enhanced - p1) / (p99 - p1)
```

**Best for**: Inconsistent CT windowing

---

### 6. Percentile Thresholding

Distribution-based adaptive thresholds.

| Percentile | Typical Use |
|------------|-------------|
| 95th | Conservative (fewer FP) |
| 99th | Standard |
| 99.5th | Aggressive (more detections) |

**Best for**: Robust outlier handling

---

### 7. Gaussian Filtering

Multi-scale smoothing for noise reduction.

```python
from scipy.ndimage import gaussian_filter

# Multi-scale Gaussian pyramid
scales = [0.5, 1.0, 1.5, 2.0]
smoothed = [gaussian_filter(image, sigma=s) for s in scales]
```

**Best for**: High-frequency noise

---

## Ensemble Generation

### Why 38 Variants?

Each enhancement method generates multiple outputs with different parameters:

| Method | Variants | Total |
|--------|----------|-------|
| CTP Enhancement | 6 | 6 |
| Wavelet | 4 | 4 |
| Adaptive Threshold | 5 | 5 |
| Morphological | 4 | 4 |
| Histogram | 5 | 5 |
| Percentile | 6 | 6 |
| Gaussian | 8 | 8 |
| **Total** | | **38** |

### Consensus Voting

```python
# Global vote map
vote_map = np.zeros_like(image)

for variant in enhanced_masks:
    vote_map += variant

# Consensus at different thresholds
consensus_50pct = vote_map >= (38 * 0.5)  # 19+ votes
top_mask_1 = vote_map >= (38 * 0.8)       # 31+ votes
```

---

## Output Files

```
Enhanced_masks/
├── enhanced_ctp_variant_1.nrrd
├── enhanced_ctp_variant_2.nrrd
├── ...
├── enhanced_wavelet_variant_1.nrrd
├── ...
└── enhanced_gaussian_variant_8.nrrd

Global_masks/
├── top_mask_1_[timestamp].nrrd      # Highest confidence
├── top_mask_2_[timestamp].nrrd      # Second-best
└── consensus_50pct_[timestamp].nrrd # 50% voting
```

---

## Technical Parameters

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `frangi_sigma_range` | 1-5 | Scale range for vesselness |
| `wavelet_level` | 3 | Decomposition depth |
| `threshold_model` | Random Forest | ML predictor |
| `morph_kernel_size` | 2-5 | Structuring element |
| `gaussian_sigma_range` | 0.5-2.0 | Smoothing scale |

### Customization

Advanced users can modify parameters in:

```
SEEG_ElectrodeLocalization/Threshold_mask/ctp_enhancer.py
```

---

## Performance

| Metric | Value |
|--------|-------|
| Processing time | ~10 minutes |
| Memory usage | ~4 GB |
| Output size | ~500 MB (38 masks) |

!!! tip "Memory Optimization"
    Enhanced masks are processed incrementally and can be deleted after voting is complete.

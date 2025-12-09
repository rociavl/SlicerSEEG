# Clinical Validation

Validation results from Hospital del Mar, Barcelona.

---

## Study Design

### Overview

| Parameter | Value |
|-----------|-------|
| **Institution** | Hospital del Mar, Barcelona, Spain |
| **Cohort Size** | 16 patients |
| **Reference Standard** | Expert manual annotations |
| **Validation Method** | Point-based DSC with 2mm tolerance |

### Methodology

**Ground Truth**: Expert-annotated electrode contact coordinates, validated by clinical neurophysiologists.

**Comparison**: Automated detections compared against ground truth using:

- **Point matching**: Detection within 2mm of ground truth = True Positive
- **Tolerance**: 2.0mm (clinically accepted localization error)

---

## Primary Results

### Detection Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
| **DSC (Overall)** | 0.569 ± 0.073 | 0.530 - 0.608 |
| **Precision** | 97.5% ± 2.2% | 96.3% - 98.7% |
| **Recall** | 68.6% ± 13.6% | 61.3% - 75.9% |
| **F1 Score** | 0.797 ± 0.098 | 0.744 - 0.850 |

### Localization Accuracy

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Mean Distance** | 0.91 ± 0.18 mm | 0.81 - 1.01 mm |
| **Median Distance** | 0.78 mm | — |
| **Within 1mm** | 72.3% | — |
| **Within 2mm** | 97.5% | — |
| **Within 3.5mm** | 97.5% | — |

### Brain Segmentation

| Metric | Value |
|--------|-------|
| **Dice Coefficient** | 0.936 ± 0.011 |

---

## Interpretation

### High Precision Design

!!! success "97.5% Precision"
    When the system detects an electrode contact, it is correct 97.5% of the time.

The algorithm prioritizes **precision over recall** by design:

| Outcome | Clinical Impact | Acceptable? |
|---------|-----------------|-------------|
| **False Positive** | Wrong neuroanatomical localization | Dangerous |
| **False Negative** | Missed contact (add manually) | Safe |

### Sub-Millimeter Accuracy

!!! info "0.91mm Mean Error"
    Average localization error is less than 1mm, well within the 2mm clinical threshold.

This exceeds typical manual annotation variability (~1-2mm).

---

## Detailed Statistics

### Per-Threshold Analysis

| Threshold | Precision | Recall | F1 | Mean Distance |
|-----------|-----------|--------|-----|---------------|
| 0.0 (raw) | 97.6% | 64.2% | 0.749 | 0.93 mm |
| 0.5 | 97.6% | 64.2% | 0.749 | 0.93 mm |
| 0.65 | 99.5% | 42.1% | 0.581 | 0.84 mm |
| 0.8 | 100% | 1.8% | 0.022 | 0.71 mm |
| **User-Selected** | **97.5%** | **68.6%** | **0.797** | **0.91 mm** |

### Clinical Accuracy Thresholds

| Distance | Percentage Within |
|----------|-------------------|
| 0.5 mm | 45.2% ± 12.1% |
| 1.0 mm | 72.3% ± 8.4% |
| 1.5 mm | 89.1% ± 5.2% |
| 2.0 mm | 97.5% ± 2.2% |
| 2.5 mm | 97.5% ± 2.2% |
| 3.0 mm | 97.5% ± 2.2% |
| 3.5 mm | 97.5% ± 2.2% |

---

## Comparison to Manual Processing

### Time Savings

| Method | Time per Case | Reduction |
|--------|---------------|-----------|
| Manual annotation | 4+ hours | — |
| SlicerSEEG | 15-30 min | **90%** |

### Workflow Impact

```
BEFORE (Manual):
├── Load CT scan
├── Identify each electrode (45-60 min each)
├── Mark each contact manually
├── Verify and correct
├── Export coordinates
└── Total: 4-6 hours

AFTER (SlicerSEEG):
├── Load CT + MRI
├── Click "Apply" (15-20 min automated)
├── Review confidence threshold
├── Reconstruct trajectories (5 min each)
├── Export coordinates
└── Total: 30-45 minutes
```

---

## Validation Methodology

### Point-Based DSC

```python
def calculate_point_based_dsc(pred_coords, gt_coords, tolerance=2.0):
    """
    DSC = 2*TP / (2*TP + FP + FN)

    Where:
    - TP = predictions within tolerance of a GT point
    - FP = predictions NOT within tolerance of any GT
    - FN = GT points NOT within tolerance of any prediction
    """
```

### Distance Metrics

For each detected contact:

1. Find nearest ground truth contact
2. Calculate Euclidean distance
3. Aggregate statistics (mean, median, percentiles)

### Excluded Cases

Some patients were excluded from validation due to:

- Incomplete ground truth annotations
- Unusual electrode configurations
- Data quality issues

---

## Reproducibility

### Validation Scripts

Available in the repository:

```
validation/
├── dsc_validation_coordinates.py    # Point-based DSC
├── trajectory_validation.py         # Trajectory validation
└── Results_cohort/                  # Raw results
```

### Running Validation

```bash
cd validation/Results_cohort/Validation_code
python dsc_validation_coordinates.py
```

Output:

- `validation_point_based_dsc.csv` - Detailed results
- `validation_point_based_summary.txt` - Summary statistics

---

## Limitations

### Known Limitations

1. **Recall vs Precision Trade-off**: Higher precision comes at cost of recall
2. **Image Quality Dependency**: Performance varies with CT quality
3. **Electrode Variability**: Different manufacturers may require parameter tuning
4. **Single-Center Validation**: Results from one institution

### Future Work

- Multi-center validation
- Different electrode manufacturers
- Real-time processing optimization
- Integration with surgical planning systems

---

## References

### Related Publications

> **Ávalos, R. (2025).** "Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science." *Bachelor's Thesis, Universitat Politècnica de Catalunya.*

### Clinical Collaboration

- Hospital del Mar Epilepsy Unit, Barcelona
- Center for Brain and Cognition, UPF

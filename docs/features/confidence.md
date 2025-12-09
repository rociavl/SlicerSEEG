# Confidence Analysis

ML-based electrode validation with 38-dimensional features.

---

## Overview

After centroid detection, each electrode candidate is evaluated using a machine learning classifier. This produces a **confidence score** (0-1) indicating how likely the candidate is a true electrode contact.

---

## Feature Extraction

### 38-Dimensional Feature Vector

Each electrode candidate is characterized by features across four categories:

=== "Spatial Features"

    | Feature | Description |
    |---------|-------------|
    | `RAS_X` | Right-Left coordinate |
    | `RAS_Y` | Anterior-Posterior coordinate |
    | `RAS_Z` | Superior-Inferior coordinate |
    | `hemisphere` | Left (-1) or Right (+1) |
    | `rel_position_X/Y/Z` | Normalized position in brain |
    | `distance_to_center` | Distance from brain centroid |

=== "Intensity Features"

    | Feature | Description |
    |---------|-------------|
    | `CT_mean` | Mean intensity at location |
    | `CT_std` | Standard deviation |
    | `CT_min` | Minimum value |
    | `CT_max` | Maximum value |
    | `CT_p25/p50/p75/p95` | Percentile values |
    | `intensity_ratio` | Relative to background |

=== "Topological Features"

    | Feature | Description |
    |---------|-------------|
    | `PCA_component_1/2/3` | Principal component scores |
    | `KDE_density` | Kernel density estimate |
    | `neighbor_dist_mean` | Mean distance to neighbors |
    | `neighbor_dist_min` | Distance to nearest neighbor |
    | `louvain_community` | Community cluster ID |
    | `n_neighbors_5mm` | Count within 5mm radius |

=== "Geometric Features"

    | Feature | Description |
    |---------|-------------|
    | `distance_to_surface` | Distance to brain surface |
    | `pixel_count` | Size of detected region |
    | `aspect_ratio` | Shape elongation |
    | `compactness` | Shape regularity |

---

## Machine Learning Model

### LightGBM Classifier

SlicerSEEG uses [LightGBM](https://lightgbm.readthedocs.io/) for confidence prediction:

- **Type**: Gradient Boosting Decision Tree
- **Training**: Patient-specific ensemble
- **Validation**: Leave-one-out cross-validation

### Patient-Specific Ensemble

To avoid overfitting to specific patient anatomy:

```python
# Leave-one-out prediction
for patient in cohort:
    # Train on all other patients
    train_data = cohort.exclude(patient)
    model = lightgbm.train(train_data)

    # Predict on held-out patient
    predictions[patient] = model.predict(patient.features)
```

---

## Confidence Scoring

### Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0 | High confidence (likely true electrode) |
| 0.5 - 0.8 | Medium confidence (review recommended) |
| 0.2 - 0.5 | Low confidence (possible false positive) |
| 0.0 - 0.2 | Very low (likely artifact) |

### Threshold Selection

The **Confidence Viewer** allows real-time threshold adjustment:

```
Higher Threshold (e.g., 0.5)
├── Fewer candidates shown
├── Higher precision (fewer false positives)
└── May miss some true electrodes

Lower Threshold (e.g., 0.05)
├── More candidates shown
├── Higher recall (fewer false negatives)
└── More false positives to review
```

!!! tip "Recommended Starting Point"
    Start with threshold = **0.05** and adjust based on visual inspection.

---

## Output Files

```
Confidence_Analysis/
├── target_features_[timestamp].csv      # 38-dim feature vectors
├── confidence_predictions_[timestamp].csv # Predicted scores
└── confidence_summary_[timestamp].txt    # Statistics
```

### Feature CSV Format

| Column | Description |
|--------|-------------|
| `point_id` | Unique identifier |
| `RAS_X`, `RAS_Y`, `RAS_Z` | Coordinates |
| `feature_1`...`feature_38` | Feature values |
| `confidence` | Predicted score |
| `prediction` | Binary (above threshold) |

---

## Interactive Viewer

### Features

- **Slider Control**: Adjust threshold in real-time
- **Statistics Display**: TP, FP, detection count
- **3D Visualization**: Color-coded by confidence
- **Export**: Save filtered coordinates

### Using the Viewer

1. After processing, the Confidence Viewer panel activates
2. Move the slider to adjust threshold
3. Observe changes in 3D view and statistics
4. When satisfied, proceed to trajectory reconstruction

---

## Performance Metrics

From clinical validation:

| Metric | Value |
|--------|-------|
| **Precision** | 97.5% ± 2.2% |
| **Recall** | 68.6% ± 13.6% |
| **F1 Score** | 0.797 ± 0.098 |
| **Mean Distance** | 0.91 ± 0.18 mm |

!!! info "Design Philosophy"
    The model prioritizes **precision over recall**. It's safer to miss contacts (which can be added manually) than to include false positives (which could mislead surgical planning).

---

## Technical Details

### Model Parameters

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
```

### Feature Importance

Top contributing features (typical):

1. `CT_mean` - Electrode metal is high intensity
2. `distance_to_surface` - Electrodes penetrate brain
3. `neighbor_dist_mean` - Contacts are regularly spaced
4. `PCA_component_1` - Linear arrangement
5. `louvain_community` - Cluster coherence

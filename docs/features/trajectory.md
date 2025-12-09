# Trajectory Reconstruction

Enhanced semi-automatic electrode contact localization.

---

## Overview

The Trajectory Reconstruction module goes beyond simple linear interpolation to provide intelligent contact localization. It combines user-defined trajectory endpoints with detected electrode contacts, automatically filling gaps and following SEEG labeling conventions.

---

## Two Reconstruction Modes

### Semi-Automatic Mode

**Best for**: High-quality CT scans with good electrode visibility

| Step | Action |
|------|--------|
| 1 | User defines entry point (skull surface) |
| 2 | User defines deepest point (brain target) |
| 3 | System finds detected contacts along trajectory |
| 4 | System fills missing contacts automatically |
| 5 | Output: Complete contact sequence |

```python
# Semi-automatic workflow
detected_contacts = electrode_predictions.get_points()
nearby = find_contacts_near_trajectory(detected_contacts, entry, deepest)
filled = fill_missing_contacts(nearby, spacing=3.5)
```

### Manual Mode

**Best for**: Challenging cases or verification

| Step | Action |
|------|--------|
| 1 | User defines entry point |
| 2 | User defines deepest point |
| 3 | System estimates contact count from length |
| 4 | System generates equally-spaced contacts |

```python
# Manual workflow
n_contacts = estimate_contact_count(entry, deepest, spacing=3.5)
contacts = linear_interpolate_contacts(entry, deepest, n_contacts)
```

---

## Intelligent Features

### Smart Contact Detection

Finds detected contacts within configurable distance threshold of the trajectory line:

```python
def find_contacts_near_trajectory(detected, entry, deepest, threshold=3.5):
    """
    For each detected contact:
    1. Calculate perpendicular distance to trajectory line
    2. If distance < threshold: include contact
    3. Sort by position along trajectory
    """
```

**Parameters**:

- `distance_threshold`: Maximum perpendicular distance (default: 3.5mm)
- Contacts beyond endpoints (±10% margin) are included

---

### Automatic Gap Filling

When contacts are missed by detection, the system estimates their positions:

```python
def fill_missing_contacts(detected, entry, deepest, spacing=3.5):
    """
    1. Generate expected positions based on spacing
    2. For each expected position:
       - If detected contact within tolerance: use detected
       - Otherwise: use interpolated position
    """
```

**Example**:
```
Expected:  1    2    3    4    5    6    7    8
Detected:  *         *    *         *         *
Filled:    D    I    D    D    I    D    I    D

D = Detected contact used
I = Interpolated (gap filled)
```

---

### SEEG Convention Labeling

Contacts are labeled following standard SEEG convention:

- **Deepest contact = 1** (closest to target structure)
- Numbers increment toward entry point
- Format: `{electrode_name}{contact_number}`

**Example**: Electrode "A" with 10 contacts
```
A1  (deepest, in target)
A2
A3
...
A10 (most superficial, near entry)
```

---

### Spline Interpolation

For curved trajectories, cubic spline fitting is available:

```python
from scipy.interpolate import splprep, splev

# Fit spline through control points
tck, u = splprep([x, y, z], s=0, k=3)

# Evaluate smooth curve
smooth_points = splev(np.linspace(0, 1, 50), tck)
```

!!! note "When to Use Spline"
    Most SEEG electrodes are straight. Use spline only if visible curvature exists.

---

### Snap-to-Point

User clicks automatically snap to nearest detected point:

```python
def snap_to_nearest_point(click, detected_points, max_distance=5.0):
    """
    If detected point within max_distance: snap to it
    Otherwise: use original click position
    """
```

---

## Visual Output

### Markup Fiducials

Each contact appears as a labeled point:

- **Node name**: `Electrode_{name}` (e.g., `Electrode_A`)
- **Point labels**: `A1`, `A2`, `A3`...
- **Visualization**: Colored spheres in 3D view

### Trajectory Curves

A line connects all contacts:

- **Node name**: `Trajectory_{name}` (e.g., `Trajectory_A`)
- **Type**: Linear curve through control points

### Hemisphere Color Coding

Colors indicate hemisphere based on R coordinate (RAS):

| Hemisphere | R Coordinate | Color |
|------------|--------------|-------|
| **Right** | R ≥ 0 | Blue (0.3, 0.5, 1.0) |
| **Left** | R < 0 | Pink (1.0, 0.4, 0.7) |

---

## Step-by-Step Workflow

### 1. Select Mode

In the **Manual Trajectory Definition** panel:

- [x] Semi-Automatic (uses `Electrode_Predictions`)
- [ ] Manual

### 2. Enter Electrode Name

Type the electrode identifier: `A`, `B`, `C`, etc.

### 3. Configure Spacing

Adjust contact spacing (default: 3.5mm)

Common values:

| Electrode Type | Typical Spacing |
|----------------|-----------------|
| Standard depth | 3.5 mm |
| High-density | 2.0 mm |
| Subdural | 5.0-10.0 mm |

### 4. Select Entry Point

1. Click **"Select Entry Point"**
2. In 3D view, click on skull surface where electrode enters
3. Point snaps to nearest detected contact if available

### 5. Select Deepest Point

1. Click **"Select Deepest Point"**
2. In 3D view, click on deepest visible contact
3. Point snaps to nearest detected contact if available

### 6. Reconstruct

Click **"Reconstruct Trajectory"**

**Output**:
```
Console: Electrode A: R=-45.2 -> Left (Pink)
Console: Semi-auto trajectory: 8 detected, 4 estimated (filled)
Console: Created markup node: Electrode_A (12 contacts) - Left (Pink)
```

### 7. Repeat

Process remaining electrodes (B, C, D, E, F, G...)

---

## Output Files

After saving the Slicer scene:

```
[patient_folder]/
├── Electrode_A.mrk.json      # Contact coordinates
├── Electrode_B.mrk.json
├── ...
├── Trajectory_A.mrk.json     # Curve nodes
├── Trajectory_B.mrk.json
└── ...
```

### Markup JSON Format

```json
{
  "markups": [{
    "type": "Fiducial",
    "coordinateSystem": "LPS",
    "controlPoints": [
      {"label": "A1", "position": [-45.2, 12.3, -8.1]},
      {"label": "A2", "position": [-43.8, 11.9, -6.2]},
      ...
    ]
  }]
}
```

---

## Validation

You can validate reconstructed trajectories against ground truth using the provided validation script:

```bash
python trajectory_validation.py
```

This calculates:

- **DSC**: Dice similarity coefficient
- **Precision/Recall/F1**: Detection metrics
- **Distance**: Mean error to ground truth contacts

See [Clinical Validation](../validation/clinical-results.md) for validation methodology.

---

## Tips for Best Results

!!! tip "Point Selection"
    - Zoom in for precise placement
    - Use orthogonal views (axial, sagittal, coronal)
    - Enable point snapping for consistency

!!! tip "Gap Filling"
    - Keep "Fill Missing Contacts" enabled for complete trajectories
    - Review filled contacts visually for accuracy

!!! tip "Multiple Electrodes"
    - Process electrodes systematically (A→B→C...)
    - Save scene periodically
    - Export each electrode after verification

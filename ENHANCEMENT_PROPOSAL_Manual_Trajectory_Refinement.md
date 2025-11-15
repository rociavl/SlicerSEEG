# Enhancement Proposal: Manual Trajectory Refinement Mode

## Overview
Combine the existing automatic trajectory reconstruction with a new manual refinement capability to give clinicians control when automatic detection needs adjustment.

---

## Current System (Automatic Path Reconstruction)

### How It Works
1. **DBSCAN Clustering**: Groups nearby electrode contacts based on spatial proximity
2. **Louvain Community Detection**: Refines clustering using graph-based community detection
3. **PCA Analysis**: Finds principal direction for each trajectory and orders contacts along the path
4. **Adaptive Clustering**: Automatically optimizes clustering parameters for best results

### Strengths
- Fully automatic - works well for ~80-90% of cases
- Handles bilateral electrode configurations
- Robust parameter optimization
- Good for well-separated, linear trajectories

### Limitations
- May fragment trajectories when contacts have irregular spacing
- Can misclassify contacts when trajectories are close together
- No easy way to correct mistakes or merge fragments
- Contacts marked as noise (-1) are lost

---

## Proposed Enhancement: Hybrid Automatic + Manual Refinement

### Core Concept
After automatic reconstruction, add a refinement tool where users can:
1. Select an existing trajectory to refine (or create a new one)
2. Manually pick **Entry Point** and **Deepest Point**
3. System automatically refits the trajectory and finds contacts along the path

### The Magic: Semi-Automatic Contact Detection
- User provides only 2 points (entry + deepest)
- System intelligently searches through ALL detected electrode contacts
- Finds contacts within a threshold distance of the fitted line
- Orders them along the trajectory direction
- Adds previously missed or misclassified contacts
- Removes outliers that don't fit the manual endpoints

---

## Workflow Integration

### Phase 1: Automatic Detection (Current)
```
Input: Electrode mask or markup points
  ↓
Run: DBSCAN + Louvain + PCA + Adaptive Clustering
  ↓
Output: Detected trajectories with automatic grouping
```

### Phase 2: Manual Refinement (NEW - Optional)
```
User reviews detected trajectories
  ↓
If trajectory needs refinement:
  1. Select trajectory to refine (or "New Trajectory")
  2. Click "Select Entry Point" → pick point in 3D view
  3. Click "Select Deepest Point" → pick point in 3D view
  4. Adjust distance threshold slider (default: 3.5mm)
  5. Click "Refit Trajectory & Update Contacts"
  ↓
System automatically:
  - Fits line between entry/deepest points
  - Searches all electrode contacts
  - Finds contacts within threshold of fitted line
  - Orders contacts along trajectory
  - Updates visualization
  ↓
Output: Refined trajectory with corrected contact grouping
```

---

## Use Cases

### 1. Rescue Fragmented Trajectories
**Problem**: Automatic clustering splits one trajectory into 2-3 fragments
**Solution**: User selects entry/deepest points spanning the full trajectory → system merges fragments

### 2. Recover Noise Points
**Problem**: Some contacts classified as noise (cluster -1) and excluded
**Solution**: Manual refinement can reclaim noise points that actually belong to a trajectory

### 3. Separate Overlapping Trajectories
**Problem**: Two close trajectories merged into one by automatic clustering
**Solution**: User manually defines entry/deepest for each trajectory → system separates them

### 4. Override Incorrect Grouping
**Problem**: Edge contacts assigned to wrong trajectory
**Solution**: User redefines trajectory endpoints → system reassigns contacts based on distance

### 5. Handle Complex Geometry
**Problem**: Curved or bent electrodes don't fit linear PCA model well
**Solution**: User-defined endpoints provide better anatomical reference

---

## Technical Implementation

### Core Function Structure

```python
def manual_trajectory_refinement(
    coords_array,              # All detected electrodes (Nx3 numpy array)
    entry_point,               # User-selected [x, y, z]
    deepest_point,             # User-selected [x, y, z]
    distance_threshold=3.5,    # mm, contacts within this distance included
    existing_trajectory_id=None,  # Optional: update existing trajectory
    dbscan_results=None,       # Optional: original clustering for reference
    exclude_clusters=None      # Optional: ignore specific clusters
):
    """
    Refit a trajectory based on manually selected entry and deepest points.

    Returns:
        refined_trajectory: dict with trajectory metrics and contact list
    """

    # 1. Calculate trajectory direction vector
    direction = (deepest_point - entry_point)
    trajectory_length = np.linalg.norm(direction)
    direction = direction / trajectory_length  # Normalize

    # 2. Find contacts within threshold distance of trajectory line
    contacts_on_path = []
    distances_to_line = []

    for coord in coords_array:
        # Point-to-line distance calculation
        vec_to_point = coord - entry_point
        projection_length = np.dot(vec_to_point, direction)
        projection_point = entry_point + projection_length * direction
        distance = np.linalg.norm(coord - projection_point)

        if distance <= distance_threshold:
            # Also check that point is between entry and deepest (not beyond)
            if 0 <= projection_length <= trajectory_length:
                contacts_on_path.append(coord)
                distances_to_line.append(distance)

    # 3. Order contacts along trajectory direction
    if len(contacts_on_path) > 0:
        contacts_array = np.array(contacts_on_path)
        projections = np.dot(contacts_array - entry_point, direction)
        sorted_indices = np.argsort(projections)
        sorted_contacts = contacts_array[sorted_indices]
    else:
        sorted_contacts = np.array([])

    # 4. Calculate trajectory metrics
    if len(sorted_contacts) >= 2:
        spacing_distances = np.linalg.norm(np.diff(sorted_contacts, axis=0), axis=1)
        avg_spacing = np.mean(spacing_distances)
        spacing_std = np.std(spacing_distances)
        spacing_regularity = spacing_std / avg_spacing if avg_spacing > 0 else np.nan

        # Linearity: how well do contacts fit the straight line?
        deviations = []
        for coord in sorted_contacts:
            vec = coord - entry_point
            proj = np.dot(vec, direction) * direction
            deviation = np.linalg.norm(vec - proj)
            deviations.append(deviation)
        avg_deviation = np.mean(deviations)
        linearity = 1.0 - (avg_deviation / trajectory_length)  # Higher is better
    else:
        avg_spacing = None
        spacing_regularity = None
        linearity = None
        avg_deviation = None

    # 5. Create refined trajectory structure
    refined_trajectory = {
        "trajectory_id": existing_trajectory_id if existing_trajectory_id is not None else "manual_refined",
        "method": "manual_refinement",
        "electrode_count": len(sorted_contacts),
        "entry_point": entry_point.tolist(),
        "deepest_point": deepest_point.tolist(),
        "direction": direction.tolist(),
        "length_mm": float(trajectory_length),
        "avg_spacing_mm": float(avg_spacing) if avg_spacing is not None else None,
        "spacing_regularity": float(spacing_regularity) if spacing_regularity is not None and not np.isnan(spacing_regularity) else None,
        "linearity": float(linearity) if linearity is not None and not np.isnan(linearity) else None,
        "avg_deviation_mm": float(avg_deviation) if avg_deviation is not None else None,
        "distance_threshold_used": distance_threshold,
        "sorted_coords": sorted_contacts.tolist(),
        "endpoints": [entry_point.tolist(), deepest_point.tolist()]
    }

    return refined_trajectory


def find_contacts_near_trajectory(coords_array, trajectory_start, trajectory_end, threshold_mm):
    """
    Helper function: Find all contacts within threshold distance of a trajectory line.
    """
    direction = trajectory_end - trajectory_start
    length = np.linalg.norm(direction)
    direction = direction / length

    nearby_contacts = []
    contact_distances = []

    for i, coord in enumerate(coords_array):
        # Calculate perpendicular distance to line
        vec = coord - trajectory_start
        proj_length = np.dot(vec, direction)
        proj_point = trajectory_start + proj_length * direction
        perp_distance = np.linalg.norm(coord - proj_point)

        # Check if within threshold and within trajectory bounds
        if perp_distance <= threshold_mm and 0 <= proj_length <= length:
            nearby_contacts.append({
                'index': i,
                'coord': coord,
                'distance_to_line': perp_distance,
                'position_along_trajectory': proj_length
            })

    return nearby_contacts


def merge_trajectory_fragments(coords_array, fragment_ids, trajectory_results):
    """
    Merge multiple detected trajectory fragments into one refined trajectory.

    Args:
        coords_array: All electrode coordinates
        fragment_ids: List of trajectory IDs to merge (e.g., [0, 2, 5])
        trajectory_results: Original trajectory detection results

    Returns:
        merged_trajectory: Combined trajectory with all contacts
    """
    # Extract all contacts from specified fragments
    all_fragment_contacts = []

    for frag_id in fragment_ids:
        traj = next((t for t in trajectory_results['trajectories'] if t['cluster_id'] == frag_id), None)
        if traj and 'sorted_coords' in traj:
            all_fragment_contacts.extend(traj['sorted_coords'])

    if len(all_fragment_contacts) < 2:
        return None

    fragment_array = np.array(all_fragment_contacts)

    # Automatically determine entry (outermost) and deepest points
    # Assume Z-axis is superior-inferior; adjust based on your coordinate system
    z_coords = fragment_array[:, 2]
    entry_idx = np.argmax(z_coords)  # Most superior
    deepest_idx = np.argmin(z_coords)  # Most inferior

    entry_point = fragment_array[entry_idx]
    deepest_point = fragment_array[deepest_idx]

    # Use manual refinement to refit merged trajectory
    merged = manual_trajectory_refinement(
        coords_array=coords_array,
        entry_point=entry_point,
        deepest_point=deepest_point,
        distance_threshold=4.0,  # Slightly higher for fragments
        existing_trajectory_id=f"merged_{'-'.join(map(str, fragment_ids))}"
    )

    merged['merged_from'] = fragment_ids

    return merged
```

---

## UI Design Proposal

### Option A: New Collapsible Section (Recommended)

Add after "2. Electrode Confidence Viewer":

```
┌─ 3. Manual Trajectory Refinement (Optional) ────────────────────┐
│                                                                  │
│ Use this tool to refine trajectories detected in Section 5.     │
│                                                                  │
│ 📊 Automatic Analysis Status:                                   │
│    • Trajectories detected: 8                                   │
│    • Total contacts: 124                                        │
│    • Noise points: 3                                            │
│                                                                  │
│ ─────────────────────────────────────────────────────────────── │
│                                                                  │
│ Refine Trajectory:                                              │
│   Trajectory ID: [Dropdown: New | 0 | 1 | 2 | 3 | 4 | 5...] │
│                                                                  │
│ Manual Endpoint Selection:                                      │
│   [🎯 Pick Entry Point]      Entry: (45.2, -12.3, 98.7)        │
│   [🎯 Pick Deepest Point]    Deepest: (42.8, -10.1, 45.2)      │
│   [↺ Clear Selections]                                          │
│                                                                  │
│ Search Parameters:                                              │
│   Distance Threshold: [━━━━●━━━━━] 3.5 mm                      │
│   (contacts within this distance will be included)              │
│                                                                  │
│ Advanced Options: [▼ Show]                                      │
│   ☐ Search noise points (cluster -1)                           │
│   ☐ Override existing trajectory contacts                      │
│   ☐ Use spline fit instead of linear                           │
│                                                                  │
│ [✨ Refit Trajectory & Find Contacts]                           │
│                                                                  │
│ Status: 🟢 Ready | ⚙️ Processing | ✅ 12 contacts found         │
│                                                                  │
│ ─────────────────────────────────────────────────────────────── │
│                                                                  │
│ Merge Trajectories:                                             │
│   Fragment IDs: [Text: e.g., 0,2,5]                            │
│   [🔗 Merge Selected Fragments]                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Option B: Integrated into Trajectory Analysis Section

Add refinement controls directly in Section 5 after trajectory analysis runs.

### Option C: Popup Dialog/Widget

Launch a dedicated refinement window with 3D view integration.

---

## Implementation Steps

### Phase 1: Core Functionality
1. Implement `manual_trajectory_refinement()` function
2. Implement `find_contacts_near_trajectory()` helper
3. Add point-to-line distance calculations
4. Test with known trajectory data

### Phase 2: UI Integration
1. Add new collapsible section to UI file
2. Create entry/deepest point selection buttons
3. Connect buttons to Slicer's interactive point placement
4. Add distance threshold slider
5. Add trajectory ID selector dropdown

### Phase 3: Slicer Integration
1. Implement 3D point picking using Slicer markup placement
2. Create visual feedback (line preview, distance sphere)
3. Update trajectory visualization after refinement
4. Save refined trajectories to output

### Phase 4: Advanced Features
1. Trajectory fragment merging tool
2. Batch refinement mode
3. Undo/redo functionality
4. Export refined trajectories to CSV

### Phase 5: Testing & Validation
1. Test with clinical datasets
2. Compare automatic vs. refined results
3. Measure time savings vs. manual placement
4. Collect clinician feedback

---

## Benefits Summary

### For Clinicians
- ✅ **Control**: Override automatic detection when needed
- ✅ **Speed**: Still faster than manual placement of all contacts
- ✅ **Accuracy**: Combine clinical knowledge with algorithmic precision
- ✅ **Recovery**: Rescue failed automatic detections without starting over
- ✅ **Flexibility**: Handle complex or unusual electrode configurations

### For the Algorithm
- ✅ **Non-destructive**: Automatic results preserved
- ✅ **Complementary**: Works alongside existing system, not replacing it
- ✅ **Reusable**: Leverages existing electrode detection and coordinates
- ✅ **Extensible**: Can add spline fitting, multi-point definition, etc.

### For Research
- ✅ **Data Quality**: More accurate trajectory definitions for studies
- ✅ **Edge Cases**: Handle difficult cases that automatic methods miss
- ✅ **Ground Truth**: Create gold-standard trajectories for algorithm training
- ✅ **Validation**: Compare automatic vs. manually refined results

---

## Future Enhancements

### Short Term
- Add trajectory smoothing/interpolation options
- Support curved electrode models (not just linear)
- Visual preview of search threshold sphere
- Contact assignment conflict resolution

### Medium Term
- Machine learning to predict optimal threshold per trajectory
- Anatomical constraints (e.g., avoid ventricles, stay in grey matter)
- Integration with electrode manufacturer specifications
- Batch processing for multiple patients

### Long Term
- Full 3D interactive trajectory editor
- Real-time refinement during electrode placement surgery
- Integration with surgical planning systems
- Automated quality metrics and validation reports

---

## Technical Considerations

### Performance
- Point-to-line distance calculation is O(n) - very fast
- Should handle 1000+ electrode contacts in real-time
- Consider spatial indexing (KD-tree) for very large datasets

### Coordinate Systems
- Ensure entry/deepest points are in same RAS coordinate system as electrodes
- Handle coordinate transformations if working with IJK or LPS
- Validate point picking returns correct world coordinates from Slicer

### Edge Cases
- What if no contacts found within threshold? → Warn user, suggest increasing threshold
- What if only 1 contact found? → Warn user, cannot define trajectory
- What if entry/deepest are very close (<5mm)? → Warn user, trajectory too short
- What if user picks points in wrong order? → Auto-detect and swap if needed

### Validation
- Check trajectory doesn't overlap with existing refined trajectories
- Warn if refined trajectory has very different spacing than expected
- Flag if linearity is low (contacts don't fit line well)
- Suggest optimal threshold based on contact distribution

---

## Related Files

- `SEEG_ElectrodeLocalization.py`: Main module file (lines 791-938: trajectory analysis)
- `construction_path.py`: Path reconstruction with DBSCAN + PCA (lines 21-159)
- `adaptive_clustering.py`: Adaptive clustering implementation
- `SEEG_ElectrodeLocalization.ui`: UI definition file

---

## Questions to Consider

1. **UI Placement**: New section vs. integrated vs. popup dialog?
2. **Point Selection**: Click in 3D view vs. select from existing markups vs. both?
3. **Threshold**: Fixed slider range vs. auto-suggest optimal?
4. **Visualization**: How to show search radius and found contacts in real-time?
5. **Undo**: Should refinement be reversible? How to store history?
6. **Export**: Save refined trajectories separately or merge with automatic results?
7. **Batch Mode**: Allow refining multiple trajectories at once?
8. **Validation**: Auto-check for quality issues after refinement?

---

## Next Steps

1. Review this proposal and gather feedback
2. Decide on UI approach (Option A, B, or C)
3. Create prototype implementation of core `manual_trajectory_refinement()` function
4. Test with sample data to validate algorithm
5. Implement UI components in Slicer
6. Run pilot testing with clinical data
7. Iterate based on clinician feedback


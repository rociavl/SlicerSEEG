"""
Trajectory Reconstruction Module for SlicerSEEG Extension

This module provides functions for manual and semi-automatic electrode trajectory
reconstruction. Users can define electrode trajectories by selecting entry and
deepest points, with options for linear interpolation or spline fitting.

Two modes are supported:
1. Semi-Automatic: Uses detected contacts from Electrode_Predictions markup
2. Manual: Estimates contact positions based on trajectory length and spacing

Author: SlicerSEEG Team
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any

try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - spline interpolation will be disabled")


# =============================================================================
# CONTACT NUMBER ESTIMATION
# =============================================================================

def estimate_contact_count(entry_point: np.ndarray,
                          deepest_point: np.ndarray,
                          contact_spacing: float = 3.5) -> Tuple[int, float]:
    """
    Estimate number of electrode contacts based on trajectory length.

    Standard SEEG electrodes have contact spacing between 3.0-5.0mm.
    Common configurations: 5, 8, 10, 12, 15, 18 contacts.

    Args:
        entry_point: [x, y, z] coordinates of entry point (skull surface)
        deepest_point: [x, y, z] coordinates of deepest point (brain target)
        contact_spacing: Expected spacing between contacts in mm (default: 3.5)

    Returns:
        Tuple of (estimated_contacts, trajectory_length_mm)
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)

    trajectory_length = np.linalg.norm(deepest_point - entry_point)

    # Estimate contacts: length / spacing + 1 (for the first contact)
    estimated_contacts = int(round(trajectory_length / contact_spacing)) + 1

    # Clamp to reasonable range (minimum 2, maximum 20)
    estimated_contacts = max(2, min(20, estimated_contacts))

    return estimated_contacts, float(trajectory_length)


# =============================================================================
# LINEAR INTERPOLATION
# =============================================================================

def linear_interpolate_contacts(entry_point: np.ndarray,
                                deepest_point: np.ndarray,
                                n_contacts: int) -> np.ndarray:
    """
    Generate equally spaced contact points along a straight line.

    Points are ordered from deepest (index 0) to entry (index n-1),
    following SEEG labeling convention where deepest contact = 1.

    Args:
        entry_point: [x, y, z] coordinates of entry point
        deepest_point: [x, y, z] coordinates of deepest point
        n_contacts: Number of contacts to generate

    Returns:
        Array of shape (n_contacts, 3) with contact coordinates,
        ordered from deepest to entry
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)

    # Generate points from deepest to entry
    t = np.linspace(0, 1, n_contacts)
    points = deepest_point + np.outer(t, entry_point - deepest_point)

    return points


# =============================================================================
# SEMI-AUTOMATIC CONTACT FITTING
# =============================================================================

def find_contacts_near_trajectory(detected_contacts: np.ndarray,
                                  entry_point: np.ndarray,
                                  deepest_point: np.ndarray,
                                  distance_threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find detected contacts that lie within a threshold distance of the trajectory line.

    This function identifies which detected electrode contacts fall along the
    user-defined trajectory path (within the specified distance threshold).

    Args:
        detected_contacts: Array of shape (N, 3) with detected contact coordinates
        entry_point: [x, y, z] coordinates of entry point
        deepest_point: [x, y, z] coordinates of deepest point
        distance_threshold: Maximum perpendicular distance from trajectory line (mm)

    Returns:
        Tuple of:
        - nearby_contacts: Coordinates of contacts near the trajectory
        - distances: Perpendicular distances to trajectory line
        - positions: Position along trajectory (0 = deepest, 1 = entry)
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)
    detected_contacts = np.array(detected_contacts)

    if len(detected_contacts) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate trajectory direction and length
    direction = entry_point - deepest_point
    trajectory_length = np.linalg.norm(direction)

    if trajectory_length < 1e-6:
        logging.warning("Entry and deepest points are too close together")
        return np.array([]), np.array([]), np.array([])

    direction_normalized = direction / trajectory_length

    nearby_contacts = []
    distances = []
    positions = []

    for contact in detected_contacts:
        # Vector from deepest point to contact
        vec_to_contact = contact - deepest_point

        # Project onto trajectory direction
        projection_length = np.dot(vec_to_contact, direction_normalized)
        projection_point = deepest_point + projection_length * direction_normalized

        # Perpendicular distance from trajectory line
        perpendicular_distance = np.linalg.norm(contact - projection_point)

        # Check if within threshold and between entry/deepest points
        # Allow small margin beyond endpoints (10% of trajectory length)
        margin = 0.1 * trajectory_length
        if (perpendicular_distance <= distance_threshold and
            -margin <= projection_length <= trajectory_length + margin):

            nearby_contacts.append(contact)
            distances.append(perpendicular_distance)
            # Normalize position to [0, 1] where 0 = deepest, 1 = entry
            positions.append(projection_length / trajectory_length)

    return (np.array(nearby_contacts),
            np.array(distances),
            np.array(positions))


def fit_trajectory_to_detected_contacts(entry_point: np.ndarray,
                                        deepest_point: np.ndarray,
                                        detected_contacts: np.ndarray,
                                        distance_threshold: float = 3.5,
                                        fit_method: str = 'linear') -> Dict[str, Any]:
    """
    Fit a trajectory through detected contacts using the specified method.

    Semi-automatic mode: User provides entry/deepest points, system finds
    and orders detected contacts that lie along this trajectory.

    Args:
        entry_point: [x, y, z] coordinates of entry point (user-selected from CT)
        deepest_point: [x, y, z] coordinates of deepest point (user-selected from CT)
        detected_contacts: Array of shape (N, 3) from Electrode_Predictions markup
        distance_threshold: Maximum distance from trajectory line to include contact
        fit_method: 'linear' or 'spline'

    Returns:
        Dictionary containing:
        - sorted_coords: Contacts ordered from deepest to entry
        - n_contacts_found: Number of detected contacts found on trajectory
        - spline_points: Fitted spline points (if fit_method='spline')
        - trajectory_length: Total length in mm
        - avg_spacing: Average spacing between contacts
        - distances_to_line: How far each contact is from the ideal line
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)

    # Find contacts near the trajectory
    nearby_contacts, distances, positions = find_contacts_near_trajectory(
        detected_contacts, entry_point, deepest_point, distance_threshold
    )

    result = {
        'method': fit_method,
        'entry_point': entry_point.tolist(),
        'deepest_point': deepest_point.tolist(),
        'trajectory_length': float(np.linalg.norm(entry_point - deepest_point)),
        'distance_threshold_used': distance_threshold,
        'n_contacts_found': len(nearby_contacts),
        'sorted_coords': [],
        'distances_to_line': [],
        'avg_spacing': None,
        'spline_points': None
    }

    if len(nearby_contacts) == 0:
        logging.warning("No detected contacts found within threshold of trajectory")
        return result

    # Sort contacts by position along trajectory (deepest to entry)
    sorted_indices = np.argsort(positions)
    sorted_coords = nearby_contacts[sorted_indices]
    sorted_distances = distances[sorted_indices]

    result['sorted_coords'] = sorted_coords.tolist()
    result['distances_to_line'] = sorted_distances.tolist()

    # Calculate average spacing
    if len(sorted_coords) >= 2:
        spacing_distances = np.linalg.norm(np.diff(sorted_coords, axis=0), axis=1)
        result['avg_spacing'] = float(np.mean(spacing_distances))

    # Spline fitting if requested and available
    if fit_method == 'spline' and SCIPY_AVAILABLE:
        spline_points, _ = spline_interpolate_contacts(
            entry_point, deepest_point, sorted_coords
        )
        result['spline_points'] = spline_points.tolist()

    return result


# =============================================================================
# SPLINE INTERPOLATION
# =============================================================================

def spline_interpolate_contacts(entry_point: np.ndarray,
                                deepest_point: np.ndarray,
                                intermediate_points: Optional[np.ndarray] = None,
                                n_output_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a smooth spline through control points.

    Control points order: deepest → intermediate_points (detected) → entry
    Entry and deepest points are anchors that the spline must pass through.

    Args:
        entry_point: [x, y, z] coordinates of entry point
        deepest_point: [x, y, z] coordinates of deepest point
        intermediate_points: Optional array of intermediate contact coordinates
        n_output_points: Number of points to sample along the spline

    Returns:
        Tuple of:
        - spline_points: Array of shape (n_output_points, 3) along the spline
        - control_points: The actual control points used for fitting
    """
    if not SCIPY_AVAILABLE:
        logging.error("scipy is required for spline interpolation")
        # Fall back to linear interpolation
        n_contacts = 2 if intermediate_points is None else len(intermediate_points) + 2
        linear_points = linear_interpolate_contacts(entry_point, deepest_point, n_output_points)
        return linear_points, np.array([deepest_point, entry_point])

    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)

    # Build control points array: deepest + intermediate + entry
    all_points = [deepest_point]

    if intermediate_points is not None and len(intermediate_points) > 0:
        intermediate_points = np.array(intermediate_points)

        # Sort intermediate points by projection along trajectory
        direction = entry_point - deepest_point
        projections = np.dot(intermediate_points - deepest_point, direction)
        sorted_idx = np.argsort(projections)

        for idx in sorted_idx:
            all_points.append(intermediate_points[idx])

    all_points.append(entry_point)
    control_points = np.array(all_points)

    # Need at least 4 points for cubic spline
    if len(control_points) < 4:
        # Use lower degree spline or linear interpolation
        if len(control_points) >= 2:
            try:
                tck, u = splprep(control_points.T, s=0, k=min(3, len(control_points)-1))
                u_new = np.linspace(0, 1, n_output_points)
                spline_points = np.array(splev(u_new, tck)).T
                return spline_points, control_points
            except Exception as e:
                logging.warning(f"Spline fitting failed: {e}, using linear interpolation")

        # Fallback to linear
        return linear_interpolate_contacts(entry_point, deepest_point, n_output_points), control_points

    try:
        # Fit cubic spline (s=0 forces through all control points)
        tck, u = splprep(control_points.T, s=0, k=3)
        u_new = np.linspace(0, 1, n_output_points)
        spline_points = np.array(splev(u_new, tck)).T
        return spline_points, control_points
    except Exception as e:
        logging.warning(f"Spline fitting failed: {e}, using linear interpolation")
        return linear_interpolate_contacts(entry_point, deepest_point, n_output_points), control_points


# =============================================================================
# FILL MISSING CONTACTS
# =============================================================================

def fill_missing_contacts(detected_contacts: np.ndarray,
                          entry_point: np.ndarray,
                          deepest_point: np.ndarray,
                          contact_spacing: float = 3.5,
                          tolerance: float = 1.5) -> Dict[str, Any]:
    """
    Fill in missing contacts along trajectory based on expected spacing.

    When using detected contacts, some may be missed by the detection algorithm.
    This function estimates expected contact positions and fills gaps where no
    detected contact exists within tolerance.

    Args:
        detected_contacts: Array of [x, y, z] detected contact positions near trajectory
        entry_point: [x, y, z] entry point
        deepest_point: [x, y, z] deepest point
        contact_spacing: Expected spacing between contacts (mm)
        tolerance: Maximum distance to consider a detected contact as "found" (mm)

    Returns:
        dict with:
            'all_contacts': List of all contact positions (detected + interpolated)
            'is_interpolated': List of booleans indicating which are interpolated
            'num_detected': Number of detected contacts matched
            'num_interpolated': Number of interpolated contacts added
    """
    entry = np.array(entry_point)
    deepest = np.array(deepest_point)
    detected = np.array(detected_contacts) if len(detected_contacts) > 0 else np.array([])

    # Calculate trajectory
    trajectory_vector = entry - deepest
    trajectory_length = np.linalg.norm(trajectory_vector)

    if trajectory_length < 1e-6:
        return {
            'all_contacts': [],
            'is_interpolated': [],
            'num_detected': 0,
            'num_interpolated': 0
        }

    trajectory_dir = trajectory_vector / trajectory_length

    # Estimate number of contacts
    num_contacts = int(round(trajectory_length / contact_spacing)) + 1

    # Generate expected positions (starting from deepest)
    expected_positions = []
    for i in range(num_contacts):
        pos = deepest + trajectory_dir * (i * contact_spacing)
        expected_positions.append(pos)

    # Match detected contacts to expected positions
    all_contacts = []
    is_interpolated = []
    detected_used = set()
    num_matched = 0

    for expected_pos in expected_positions:
        # Find closest detected contact not yet used
        min_dist = float('inf')
        closest_idx = None

        for idx in range(len(detected)):
            if idx in detected_used:
                continue
            dist = np.linalg.norm(detected[idx] - expected_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        # If within tolerance, use detected contact
        if closest_idx is not None and min_dist <= tolerance:
            all_contacts.append(detected[closest_idx].tolist())
            is_interpolated.append(False)
            detected_used.add(closest_idx)
            num_matched += 1
        else:
            # Use interpolated position
            all_contacts.append(expected_pos.tolist())
            is_interpolated.append(True)

    return {
        'all_contacts': all_contacts,
        'is_interpolated': is_interpolated,
        'num_detected': num_matched,
        'num_interpolated': sum(is_interpolated)
    }


# =============================================================================
# CONTACT LABELING (SEEG CONVENTION)
# =============================================================================

def label_contacts_seeg_convention(sorted_coords: np.ndarray,
                                   electrode_name: str = "A") -> List[str]:
    """
    Label contacts following SEEG convention: deepest contact = 1.

    Standard SEEG labeling has the deepest contact (closest to target structure)
    as contact 1, with numbers incrementing toward the entry point.

    Args:
        sorted_coords: Array of shape (N, 3) ordered from deepest to entry
        electrode_name: Electrode name/prefix (e.g., "A", "LAH", "RMH")

    Returns:
        List of labels: ["{electrode_name}1", "{electrode_name}2", ...]
    """
    n_contacts = len(sorted_coords)
    labels = [f"{electrode_name}{i+1}" for i in range(n_contacts)]
    return labels


# =============================================================================
# SNAP TO NEAREST POINT
# =============================================================================

def snap_to_nearest_point(click_position: np.ndarray,
                          detected_points: np.ndarray,
                          max_distance: float = 5.0) -> Tuple[np.ndarray, bool, float]:
    """
    Snap a clicked position to the nearest detected point if within threshold.

    Args:
        click_position: [x, y, z] coordinates where user clicked
        detected_points: Array of shape (N, 3) of available points to snap to
        max_distance: Maximum distance to snap (mm)

    Returns:
        Tuple of:
        - snapped_position: The position (snapped or original)
        - was_snapped: Boolean indicating if snapping occurred
        - snap_distance: Distance to snapped point (or inf if not snapped)
    """
    click_position = np.array(click_position)

    if detected_points is None or len(detected_points) == 0:
        return click_position, False, float('inf')

    detected_points = np.array(detected_points)
    distances = np.linalg.norm(detected_points - click_position, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    if min_distance <= max_distance:
        return detected_points[min_idx], True, float(min_distance)

    return click_position, False, float(min_distance)


# =============================================================================
# TRAJECTORY OUTPUT GENERATION
# =============================================================================

def create_trajectory_output(entry_point: np.ndarray,
                            deepest_point: np.ndarray,
                            sorted_coords: np.ndarray,
                            electrode_name: str,
                            method: str,
                            spline_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create a standardized trajectory output dictionary.

    This format is compatible with the existing trajectory analysis output
    from adaptive_clustering.py.

    Args:
        entry_point: Entry point coordinates
        deepest_point: Deepest point coordinates
        sorted_coords: Ordered contact coordinates (deepest to entry)
        electrode_name: Electrode identifier
        method: "manual_estimated" or "manual_semi_auto"
        spline_points: Optional spline curve points

    Returns:
        Dictionary with trajectory information
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)
    sorted_coords = np.array(sorted_coords)

    trajectory_length = np.linalg.norm(entry_point - deepest_point)

    # Calculate spacing
    avg_spacing = None
    spacing_regularity = None
    if len(sorted_coords) >= 2:
        spacing_distances = np.linalg.norm(np.diff(sorted_coords, axis=0), axis=1)
        avg_spacing = float(np.mean(spacing_distances))
        if len(spacing_distances) > 1:
            spacing_regularity = float(np.std(spacing_distances) / avg_spacing)

    # Generate labels
    labels = label_contacts_seeg_convention(sorted_coords, electrode_name)

    # Calculate direction vector
    direction = (entry_point - deepest_point) / trajectory_length if trajectory_length > 0 else [0, 0, 0]

    output = {
        "trajectory_id": electrode_name,
        "method": method,
        "electrode_name": electrode_name,
        "electrode_count": len(sorted_coords),
        "entry_point": entry_point.tolist(),
        "deepest_point": deepest_point.tolist(),
        "direction": direction.tolist() if isinstance(direction, np.ndarray) else direction,
        "length_mm": float(trajectory_length),
        "avg_spacing_mm": avg_spacing,
        "spacing_regularity": spacing_regularity,
        "sorted_coords": sorted_coords.tolist(),
        "labels": labels,
        "endpoints": [deepest_point.tolist(), entry_point.tolist()],
        "spline_points": spline_points.tolist() if spline_points is not None else None
    }

    return output


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_trajectory_points(entry_point: np.ndarray,
                              deepest_point: np.ndarray,
                              min_length: float = 5.0) -> Tuple[bool, str]:
    """
    Validate that entry and deepest points form a valid trajectory.

    Args:
        entry_point: Entry point coordinates
        deepest_point: Deepest point coordinates
        min_length: Minimum trajectory length in mm

    Returns:
        Tuple of (is_valid, error_message)
    """
    entry_point = np.array(entry_point)
    deepest_point = np.array(deepest_point)

    # Check for NaN/Inf
    if not np.all(np.isfinite(entry_point)) or not np.all(np.isfinite(deepest_point)):
        return False, "Entry or deepest point contains invalid coordinates (NaN/Inf)"

    # Check minimum length
    trajectory_length = np.linalg.norm(entry_point - deepest_point)
    if trajectory_length < min_length:
        return False, f"Trajectory too short ({trajectory_length:.1f}mm < {min_length}mm minimum)"

    return True, ""


def get_point_from_markup_node(markup_node, point_index: int = 0) -> Optional[np.ndarray]:
    """
    Extract a point coordinate from a Slicer markup node.

    Args:
        markup_node: vtkMRMLMarkupsFiducialNode
        point_index: Index of the point to extract

    Returns:
        numpy array [x, y, z] or None if not available
    """
    if markup_node is None:
        return None

    if point_index >= markup_node.GetNumberOfControlPoints():
        return None

    point = [0.0, 0.0, 0.0]
    markup_node.GetNthControlPointPosition(point_index, point)
    return np.array(point)


def get_all_points_from_markup_node(markup_node) -> Optional[np.ndarray]:
    """
    Extract all point coordinates from a Slicer markup node.

    Args:
        markup_node: vtkMRMLMarkupsFiducialNode

    Returns:
        numpy array of shape (N, 3) or None if not available
    """
    if markup_node is None:
        return None

    n_points = markup_node.GetNumberOfControlPoints()
    if n_points == 0:
        return None

    points = []
    for i in range(n_points):
        point = [0.0, 0.0, 0.0]
        markup_node.GetNthControlPointPosition(i, point)
        points.append(point)

    return np.array(points)

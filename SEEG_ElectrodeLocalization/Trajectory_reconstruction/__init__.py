"""
Trajectory Reconstruction Module for SlicerSEEG

This module provides manual and semi-automatic electrode trajectory reconstruction
capabilities for the SEEG electrode localization workflow.
"""

from .trajectory_reconstruction import (
    # Contact estimation
    estimate_contact_count,

    # Interpolation
    linear_interpolate_contacts,
    spline_interpolate_contacts,

    # Semi-automatic fitting
    find_contacts_near_trajectory,
    fit_trajectory_to_detected_contacts,

    # Fill missing contacts
    fill_missing_contacts,

    # Labeling
    label_contacts_seeg_convention,

    # Snap functionality
    snap_to_nearest_point,

    # Output generation
    create_trajectory_output,

    # Utilities
    validate_trajectory_points,
    get_point_from_markup_node,
    get_all_points_from_markup_node,
)

__all__ = [
    'estimate_contact_count',
    'linear_interpolate_contacts',
    'spline_interpolate_contacts',
    'find_contacts_near_trajectory',
    'fit_trajectory_to_detected_contacts',
    'fill_missing_contacts',
    'label_contacts_seeg_convention',
    'snap_to_nearest_point',
    'create_trajectory_output',
    'validate_trajectory_points',
    'get_point_from_markup_node',
    'get_all_points_from_markup_node',
]

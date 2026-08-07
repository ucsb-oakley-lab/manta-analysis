"""Underwater stereo-calibration teaching utilities."""

from .core import (
    calibrate_stereo,
    detect_checkerboard_pairs,
    detect_checkerboard_pairs_side_by_side,
    load_detection_cache,
    load_legacy_opencv_maps,
    read_side_by_side_pair,
    rectify_pair,
    rectify_videos,
    save_calibration,
    save_detection_cache,
    select_calibration_pairs,
    split_side_by_side_frame,
    vertical_disparity,
)

__all__ = [
    "calibrate_stereo",
    "detect_checkerboard_pairs",
    "detect_checkerboard_pairs_side_by_side",
    "load_detection_cache",
    "load_legacy_opencv_maps",
    "read_side_by_side_pair",
    "rectify_pair",
    "rectify_videos",
    "save_calibration",
    "save_detection_cache",
    "select_calibration_pairs",
    "split_side_by_side_frame",
    "vertical_disparity",
]

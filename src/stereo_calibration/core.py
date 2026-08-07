"""Core stereo checkerboard detection, calibration, validation, and rectification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

DETECTION_CACHE_VERSION = 3


def _open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def _board_metrics(corners: np.ndarray, width: int, height: int) -> dict[str, float]:
    xy = corners.reshape(-1, 2)
    center = xy.mean(axis=0)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    # Median adjacent-corner distance is a robust proxy for apparent board size.
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "board_scale": float(np.median(distances)),
        "border_clearance": float(min(xmin, ymin, width - xmax, height - ymax)),
    }


def _movement_metrics(
    center: np.ndarray,
    previous_center: np.ndarray | None,
    frame_number: int,
    previous_frame: int | None,
) -> tuple[float, float, float]:
    """Return displacement, displacement/frame, and elapsed frames."""
    if previous_center is None or previous_frame is None:
        return np.nan, np.nan, np.nan
    elapsed_frames = frame_number - previous_frame
    if elapsed_frames <= 0:
        raise ValueError("Detection frame numbers must increase")
    displacement = float(np.linalg.norm(center - previous_center))
    return displacement, displacement / elapsed_frames, float(elapsed_frames)


def detect_checkerboard_pairs(
    left_video: str | Path,
    right_video: str | Path,
    checkerboard_size: tuple[int, int],
    *,
    frame_offset: int = 0,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Find synchronized frames where the checkerboard is visible in both cameras.

    `frame_offset` is added to the right-camera frame number. Refined corner arrays
    are returned separately so the table remains easy to inspect and save.
    """
    left = _open_video(left_video)
    right = _open_video(right_video)
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_size = (
        int(right.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(right.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if right_size != (width, height):
        raise ValueError(f"Video sizes differ: left={(width, height)}, right={right_size}")

    total = min(
        int(left.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(right.get(cv2.CAP_PROP_FRAME_COUNT)) - max(frame_offset, 0),
    )
    stop = total if end_frame is None else min(end_frame, total)
    rows: list[dict[str, float | int]] = []
    corner_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    previous_center: np.ndarray | None = None
    previous_frame: int | None = None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for frame_number in range(start_frame, stop, frame_step):
        right_number = frame_number + frame_offset
        if right_number < 0:
            continue
        left.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        right.set(cv2.CAP_PROP_POS_FRAMES, right_number)
        ok_l, image_l = left.read()
        ok_r, image_r = right.read()
        if not (ok_l and ok_r):
            continue
        gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
        found_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_size, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_size, flags)
        if not (found_l and found_r):
            continue
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        metrics_l = _board_metrics(corners_l, width, height)
        metrics_r = _board_metrics(corners_r, width, height)
        pair_center = np.array([
            metrics_l["center_x"], metrics_l["center_y"],
            metrics_r["center_x"], metrics_r["center_y"],
        ])
        movement, movement_per_frame, detection_gap = _movement_metrics(
            pair_center, previous_center, frame_number, previous_frame
        )
        previous_center = pair_center
        previous_frame = frame_number
        rows.append({
            "frame_left": frame_number,
            "frame_right": right_number,
            "board_scale": min(metrics_l["board_scale"], metrics_r["board_scale"]),
            "border_clearance": min(metrics_l["border_clearance"], metrics_r["border_clearance"]),
            "movement": movement,
            "movement_per_frame": movement_per_frame,
            "detection_gap": detection_gap,
            "center_x": (metrics_l["center_x"] + metrics_r["center_x"]) / 2,
            "center_y": (metrics_l["center_y"] + metrics_r["center_y"]) / 2,
            "center_left_x": metrics_l["center_x"],
            "center_left_y": metrics_l["center_y"],
            "center_right_x": metrics_r["center_x"],
            "center_right_y": metrics_r["center_y"],
        })
        corner_pairs[frame_number] = (corners_l, corners_r)

    left.release()
    right.release()
    return pd.DataFrame(rows), corner_pairs


def split_side_by_side_frame(
    frame: np.ndarray,
    *,
    layout: str = "left_right",
) -> tuple[np.ndarray, np.ndarray]:
    """Split one even-width stereo frame into logical left and right views."""
    if frame.ndim < 2 or frame.shape[1] % 2:
        raise ValueError(f"Side-by-side frame width must be even; got shape {frame.shape}")
    midpoint = frame.shape[1] // 2
    first, second = frame[:, :midpoint], frame[:, midpoint:]
    if layout == "left_right":
        return first, second
    if layout == "right_left":
        return second, first
    raise ValueError("layout must be 'left_right' or 'right_left'")


def read_side_by_side_pair(
    video: str | Path,
    frame_number: int,
    *,
    layout: str = "left_right",
) -> tuple[np.ndarray, np.ndarray]:
    """Read and split a numbered side-by-side stereo frame."""
    capture = _open_video(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video}")
    return split_side_by_side_frame(frame, layout=layout)


def detect_checkerboard_pairs_side_by_side(
    video: str | Path,
    checkerboard_size: tuple[int, int],
    *,
    layout: str = "left_right",
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    detection_scale: float = 0.5,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Find paired checkerboards in a single synchronized side-by-side video."""
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not 0 < detection_scale <= 1:
        raise ValueError("detection_scale must be greater than 0 and at most 1")
    capture = _open_video(video)
    full_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_width % 2:
        capture.release()
        raise ValueError(f"Side-by-side video width must be even; got {full_width}")
    width = full_width // 2
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = total if end_frame is None else min(end_frame, total)
    rows: list[dict[str, float | int]] = []
    corner_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    previous_center: np.ndarray | None = None
    previous_frame: int | None = None
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in range(start_frame, stop):
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_number - start_frame) % frame_step:
            continue
        image_l, image_r = split_side_by_side_frame(frame, layout=layout)
        gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
        if detection_scale < 1:
            search_l = cv2.resize(gray_l, None, fx=detection_scale, fy=detection_scale)
            search_r = cv2.resize(gray_r, None, fx=detection_scale, fy=detection_scale)
        else:
            search_l, search_r = gray_l, gray_r
        found_l, corners_l = cv2.findChessboardCorners(search_l, checkerboard_size, flags)
        found_r, corners_r = cv2.findChessboardCorners(search_r, checkerboard_size, flags)
        if not (found_l and found_r):
            continue
        corners_l = corners_l / detection_scale
        corners_r = corners_r / detection_scale
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        metrics_l = _board_metrics(corners_l, width, height)
        metrics_r = _board_metrics(corners_r, width, height)
        pair_center = np.array([
            metrics_l["center_x"], metrics_l["center_y"],
            metrics_r["center_x"], metrics_r["center_y"],
        ])
        movement, movement_per_frame, detection_gap = _movement_metrics(
            pair_center, previous_center, frame_number, previous_frame
        )
        previous_center = pair_center
        previous_frame = frame_number
        rows.append({
            "frame_left": frame_number,
            "frame_right": frame_number,
            "board_scale": min(metrics_l["board_scale"], metrics_r["board_scale"]),
            "border_clearance": min(metrics_l["border_clearance"], metrics_r["border_clearance"]),
            "movement": movement,
            "movement_per_frame": movement_per_frame,
            "detection_gap": detection_gap,
            "center_x": (metrics_l["center_x"] + metrics_r["center_x"]) / 2,
            "center_y": (metrics_l["center_y"] + metrics_r["center_y"]) / 2,
            "center_left_x": metrics_l["center_x"],
            "center_left_y": metrics_l["center_y"],
            "center_right_x": metrics_r["center_x"],
            "center_right_y": metrics_r["center_y"],
        })
        corner_pairs[frame_number] = (corners_l, corners_r)

    capture.release()
    return pd.DataFrame(rows), corner_pairs


def select_calibration_pairs(
    detections: pd.DataFrame,
    *,
    number: int = 30,
    min_board_scale: float = 20,
    min_border_clearance: float = 20,
    max_movement: float = 10,
) -> pd.DataFrame:
    """Filter poor detections and greedily select diverse board poses.

    Diversity is measured over normalized image position and apparent board size.
    """
    required = {"board_scale", "border_clearance", "movement", "center_x", "center_y"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing detection columns: {sorted(missing)}")
    candidates = detections[
        (detections.board_scale >= min_board_scale)
        & (detections.border_clearance >= min_border_clearance)
        & (detections.movement.fillna(0) <= max_movement)
    ].copy()
    if candidates.empty:
        raise ValueError("No detections passed the selection thresholds.")
    # Average center avoids double-weighting two highly correlated camera views.
    # Separate centers remain in the table for camera-specific coverage plots.
    features = candidates[["center_x", "center_y", "board_scale"]].to_numpy(float)
    span = np.ptp(features, axis=0)
    features = (features - features.min(axis=0)) / np.where(span == 0, 1, span)
    chosen = [int(np.argmax(candidates.board_scale.to_numpy()))]
    while len(chosen) < min(number, len(candidates)):
        remaining = [i for i in range(len(candidates)) if i not in chosen]
        distance_to_chosen = np.linalg.norm(
            features[remaining, None, :] - features[np.array(chosen)[None, :], :], axis=2
        )
        chosen.append(remaining[int(np.argmax(distance_to_chosen.min(axis=1)))])
    return candidates.iloc[chosen].sort_values("frame_left").reset_index(drop=True)


def _object_points(checkerboard_size: tuple[int, int], square_size: float) -> np.ndarray:
    points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    points[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)
    return points * square_size


def calibrate_stereo(
    selected: pd.DataFrame,
    corner_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    checkerboard_size: tuple[int, int],
    square_size: float,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray | float]:
    """Estimate intrinsics, stereo geometry, and rectification maps."""
    object_template = _object_points(checkerboard_size, square_size)
    object_points, left_points, right_points = [], [], []
    for frame in selected.frame_left.astype(int):
        if frame not in corner_pairs:
            raise KeyError(f"Corners unavailable for selected frame {frame}")
        left_corners, right_corners = corner_pairs[frame]
        object_points.append(object_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)

    rms_l, matrix_l, distortion_l, _, _ = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None
    )
    rms_r, matrix_r, distortion_r, _, _ = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None
    )
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, matrix_l, distortion_l, matrix_r, distortion_r, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        matrix_l,
        distortion_l,
        matrix_r,
        distortion_r,
        image_size,
        criteria=SUBPIX_CRITERIA,
        flags=stereo_flags,
    )
    rect_l, rect_r, proj_l, proj_r, q_matrix, _, _ = cv2.stereoRectify(
        matrix_l, distortion_l, matrix_r, distortion_r, image_size, rotation, translation, alpha=0
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        matrix_l, distortion_l, rect_l, proj_l, image_size, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        matrix_r, distortion_r, rect_r, proj_r, image_size, cv2.CV_32FC1
    )
    return {
        "rms_left": float(rms_l), "rms_right": float(rms_r), "rms_stereo": float(rms_stereo),
        "matrix_left": matrix_l, "distortion_left": distortion_l,
        "matrix_right": matrix_r, "distortion_right": distortion_r,
        "rotation": rotation, "translation": translation,
        "essential": essential, "fundamental": fundamental,
        "rectification_left": rect_l, "rectification_right": rect_r,
        "projection_left": proj_l, "projection_right": proj_r, "q_matrix": q_matrix,
        "map_left_x": map_lx, "map_left_y": map_ly,
        "map_right_x": map_rx, "map_right_y": map_ry,
        "image_size": np.asarray(image_size),
    }


def save_calibration(path: str | Path, calibration: dict[str, np.ndarray | float]) -> None:
    """Save calibration arrays and scalar errors without Python pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **calibration)


def save_detection_cache(
    path: str | Path,
    detections: pd.DataFrame,
    corner_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save checkerboard detections and refined corners without Python pickle."""
    frames = np.asarray(sorted(corner_pairs), dtype=np.int64)
    if len(frames) == 0:
        raise ValueError("Cannot cache an empty corner-pair collection.")
    payload: dict[str, np.ndarray] = {
        "cache_version": np.asarray(DETECTION_CACHE_VERSION, dtype=np.int64),
        "frames": frames,
        "corners_left": np.stack([corner_pairs[int(frame)][0] for frame in frames]),
        "corners_right": np.stack([corner_pairs[int(frame)][1] for frame in frames]),
        "detection_columns_json": np.asarray(json.dumps(list(detections.columns))),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    for column in detections.columns:
        payload[f"detection__{column}"] = detections[column].to_numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_detection_cache(
    path: str | Path,
) -> tuple[
    pd.DataFrame,
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, object],
]:
    """Load a cache written by :func:`save_detection_cache`."""
    with np.load(path, allow_pickle=False) as archive:
        cached_version = int(archive["cache_version"]) if "cache_version" in archive else 1
        if cached_version != DETECTION_CACHE_VERSION:
            raise ValueError(
                f"Detection cache version {cached_version} is incompatible with "
                f"version {DETECTION_CACHE_VERSION}; regenerate detections."
            )
        columns = json.loads(str(archive["detection_columns_json"]))
        detections = pd.DataFrame({
            column: archive[f"detection__{column}"]
            for column in columns
        })
        frames = archive["frames"].astype(int)
        corners_left = archive["corners_left"]
        corners_right = archive["corners_right"]
        metadata = json.loads(str(archive["metadata_json"]))
    corner_pairs = {
        int(frame): (left, right)
        for frame, left, right in zip(frames, corners_left, corners_right)
    }
    return detections, corner_pairs, metadata


def load_legacy_opencv_maps(
    path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load rectification maps written by OpenCV ``FileStorage``.

    This adapter is primarily for reproducing older WALL-E analyses. New MANTA
    calibrations should be saved as ``.npz`` files with :func:`save_calibration`.
    """
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open OpenCV calibration file: {path}")
    try:
        calibration = {
            "map_left_x": storage.getNode("stereoMapL_x").mat(),
            "map_left_y": storage.getNode("stereoMapL_y").mat(),
            "map_right_x": storage.getNode("stereoMapR_x").mat(),
            "map_right_y": storage.getNode("stereoMapR_y").mat(),
        }
    finally:
        storage.release()
    if any(value is None for value in calibration.values()):
        raise ValueError(f"One or more stereo maps are missing from {path}")
    if image_size is None:
        height, width = calibration["map_left_x"].shape[:2]
        image_size = (width, height)
    calibration["image_size"] = np.asarray(image_size)
    return calibration


def rectify_pair(
    image_left: np.ndarray,
    image_right: np.ndarray,
    calibration: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.remap(image_left, calibration["map_left_x"], calibration["map_left_y"], cv2.INTER_LINEAR)
    right = cv2.remap(image_right, calibration["map_right_x"], calibration["map_right_y"], cv2.INTER_LINEAR)
    return left, right


def vertical_disparity(
    corners_left: np.ndarray,
    corners_right: np.ndarray,
    calibration: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Return absolute vertical disparities after rectifying detected corners."""
    left = cv2.undistortPoints(
        corners_left, calibration["matrix_left"], calibration["distortion_left"],
        R=calibration["rectification_left"], P=calibration["projection_left"],
    )
    right = cv2.undistortPoints(
        corners_right, calibration["matrix_right"], calibration["distortion_right"],
        R=calibration["rectification_right"], P=calibration["projection_right"],
    )
    return np.abs(left[:, 0, 1] - right[:, 0, 1])


def rectify_videos(
    left_video: str | Path,
    right_video: str | Path,
    output_left: str | Path,
    output_right: str | Path,
    calibration: dict[str, np.ndarray | float],
    *,
    frame_offset: int = 0,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    """Rectify synchronized videos using a fixed whole-frame offset."""
    left = _open_video(left_video)
    right = _open_video(right_video)
    left.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    right.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if frame_offset > 0:
        right.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    elif frame_offset < 0:
        left.set(cv2.CAP_PROP_POS_FRAMES, -frame_offset)
    width, height = map(int, np.asarray(calibration["image_size"]))
    fps = left.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_l = cv2.VideoWriter(str(output_left), fourcc, fps, (width, height))
    out_r = cv2.VideoWriter(str(output_right), fourcc, fps, (width, height))
    if not (out_l.isOpened() and out_r.isOpened()):
        raise RuntimeError("Could not open one or both output videos for writing")
    frames_written = 0
    while True:
        if end_frame is not None and start_frame + frames_written >= end_frame:
            break
        ok_l, image_l = left.read()
        ok_r, image_r = right.read()
        if not (ok_l and ok_r):
            break
        rect_l, rect_r = rectify_pair(image_l, image_r, calibration)
        out_l.write(rect_l)
        out_r.write(rect_r)
        frames_written += 1
    left.release(); right.release(); out_l.release(); out_r.release()
    return frames_written

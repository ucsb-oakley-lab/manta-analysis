#!/usr/bin/env python3
"""Quickly scan a folder of videos for bright bioluminescent blobs/signals.

This script reads directly from a local mounted SMB path (for example:
/Volumes/oakley_lab/...)
without copying files to the local machine. It:
  - walks all video files under the target directory
  - samples frames at a fixed rate to keep runtime manageable
  - thresholds each sampled frame to detect stationary or moving bright blobs
  - records each accepted contour independently with its position and intensity
  - skips frames whose average brightness indicates that the lights are on
  - writes frame-level contour data to CSV

Typical usage:
    python scan_bioluminescence.py \
        "/Volumes/oakley_lab/aquarium/2026Pan/20261111" \
        --output /Users/oakley/temporary/biolum_results

If the SMB share is not mounted locally, mount it first, e.g. on macOS:
    mkdir -p /Volumes/oakley_lab
    mount_smbfs //username:password@oakley-nas1/oakley_lab /Volumes/oakley_lab
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd

from nightly_bioluminescence import default_nas_output_dir

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".wmv", ".m4v", ".ts"}
TIMESTAMP_RE = re.compile(r"20\d{6}T\d{6}Z")
LEGACY_NUMBER_RE = re.compile(r"(?:output|video)[_-]?(\d+)", re.IGNORECASE)
DETECTION_FIELDS = [
    "night", "video", "frame", "time_sec", "x", "y", "y_flipped", "area",
    "min_intensity", "max_intensity", "mean_intensity", "bbox_x", "bbox_y",
    "bbox_width", "bbox_height",
]


def extract_night(video_path: Path) -> str:
    """Return the YYYYMMDD night associated with a video path.

    The night is normally the parent directory (for example ``20261116``).
    Timestamped camera filenames are also accepted, so a standalone file such
    as ``20261116_231500.mp4`` still gets a useful night value.
    """
    for part in (video_path.parent.name, video_path.stem):
        match = re.search(r"(?<!\d)(20\d{6})(?!\d)", part)
        if match:
            return match.group(1)
    return video_path.parent.name


def normalize_input_path(raw_path: str) -> str:
    """Resolve a mounted SMB path or a smb:// URL into a filesystem path."""
    path = raw_path.strip()
    if not path:
        raise ValueError("Input path is empty")

    if path.startswith("smb://"):
        parsed = urlparse(path)
        if parsed.netloc:
            candidate = Path("/Volumes") / parsed.path.lstrip("/")
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"SMB path '{path}' is not mounted locally. "
            "Mount the share first, then point this script at the mounted folder."
        )

    p = Path(path)
    if p.exists():
        return str(p)

    # Some users may pass a path like '~/...' or a symlink that expands
    expanded = Path(path).expanduser()
    if expanded.exists():
        return str(expanded)

    raise FileNotFoundError(f"Input path does not exist: {path}")


def find_videos(root: Path) -> List[Path]:
    videos: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return sorted(videos, key=video_sort_key)


def video_sort_key(path: Path) -> Tuple[int, Any]:
    """Sort timestamped files chronologically and legacy files numerically."""
    timestamp = TIMESTAMP_RE.search(path.name)
    if timestamp:
        return (0, timestamp.group(0))
    legacy = LEGACY_NUMBER_RE.search(path.stem)
    if legacy:
        return (1, int(legacy.group(1)))
    return (2, path.name)


def make_timestamp_mask(frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """Return a binary mask that hides the top-left timestamp region."""
    h, w = frame_shape[:2]
    # This camera's timestamp spans much of the upper-left/top portion of the frame.
    # Mask the full text band so individual digits cannot become detections.
    mask = np.ones((h, w), dtype=np.uint8) * 255
    ts_w = min(w, max(140, int(w * 0.50)))
    ts_h = min(h, max(50, int(h * 0.12)))
    mask[:ts_h, :ts_w] = 0
    return mask


def detect_blob_pixels(frame_gray: np.ndarray, min_area: int, max_area: int) -> List[Dict[str, float]]:
    """Threshold motion/background-subtracted image and return detected blob stats."""
    # Keep it conservative so we don't waste time on tiny noise.
    _, thresh = cv2.threshold(frame_gray, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Dict[str, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append({
            "area": float(area),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
        })
    return detections


def group_detections(detections: List[Dict[str, Any]], fps: float, max_gap_sec: float) -> List[List[Dict[str, Any]]]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["frame_idx"])
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    prev_time_sec = None

    for det in detections:
        t = det["frame_idx"] / fps
        if not current:
            current = [det]
            prev_time_sec = t
            continue

        if t - prev_time_sec <= max_gap_sec:
            current.append(det)
        else:
            groups.append(current)
            current = [det]
        prev_time_sec = t

    if current:
        groups.append(current)

    return groups


def estimate_pulse_count(event: List[Dict[str, Any]]) -> int:
    if len(event) < 2:
        return 1

    values = np.array([d["brightness"] for d in event], dtype=np.float32)
    if values.size < 3:
        return int(np.count_nonzero(values > np.median(values)))

    smoothed = np.convolve(values, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    baseline = np.median(smoothed)
    spread = max(np.std(smoothed), 1.0)
    threshold = baseline + 0.15 * spread

    peaks = []
    for i in range(1, smoothed.size - 1):
        if smoothed[i] > threshold and smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks.append(i)

    # In practice the raw frame detections can oversample the same event; remove tiny dupes
    # by ensuring peaks are separated by at least 2 samples.
    deduped: List[int] = []
    for idx in peaks:
        if not deduped or idx - deduped[-1] >= 2:
            deduped.append(idx)
    return len(deduped)


def scan_video(video_path: Path, sample_rate_hz: float, min_area: int, max_area: int, min_signal_frames: int,
               max_gap_sec: float, output_dir: Path, save_thumbs: bool,
               roi: Optional[Tuple[float, float, float, float]] = None,
               brightness_threshold: int = 180,
               start_sec: Optional[float] = None,
               end_sec: Optional[float] = None) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    if fps <= 0:
        fps = 10.0

    frame_step = max(1, int(round(fps / sample_rate_hz)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    first_frame = max(0, int((start_sec or 0.0) * fps))
    last_frame = None if end_sec is None else max(first_frame, int(end_sec * fps))
    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    detections: List[Dict[str, Any]] = []
    frame_idx = first_frame

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if last_frame is not None and frame_idx > last_frame:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # Resize only if the video is huge; this keeps CPU use down.
        working = frame
        if max(working.shape[:2]) > 1200:
            scale = 1200.0 / max(working.shape[:2])
            working = cv2.resize(working, (0, 0), fx=scale, fy=scale)
        scale_x = frame.shape[1] / working.shape[1]
        scale_y = frame.shape[0] / working.shape[0]

        # Detect bright pixels directly. This works for stationary signals that last
        # several seconds and does not depend on background subtraction.
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.inRange(gray, brightness_threshold, 255)

        # Hide the upper-left timestamp from the detector. It is static and should not be counted.
        timestamp_mask = make_timestamp_mask(working.shape)
        if roi is not None:
            roi_x0, roi_y0, roi_x1, roi_y1 = roi
            h, w = working.shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            x0 = max(0, min(w, int(w * roi_x0)))
            y0 = max(0, min(h, int(h * roi_y0)))
            x1 = max(x0, min(w, int(w * roi_x1)))
            y1 = max(y0, min(h, int(h * roi_y1)))
            roi_mask[y0:y1, x0:x1] = 255
            timestamp_mask = cv2.bitwise_and(timestamp_mask, roi_mask)
        bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=timestamp_mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        # Keep only bright candidate regions within the requested size range.
        blobs = detect_blob_pixels(bright_mask, min_area=min_area, max_area=max_area)
        for blob in blobs:
            x = int(blob["x"])
            y = int(blob["y"])
            w = int(blob["w"])
            h = int(blob["h"])
            crop = working[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            brightness = float(np.mean(crop))
            detections.append({
                "frame_idx": frame_idx,
                "brightness": brightness,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "time_sec": frame_idx / fps,
                "video_path": str(video_path),
            })

        frame_idx += 1

    cap.release()

    # Group frame-level blobs into candidate events.
    groups = group_detections(detections, fps=fps, max_gap_sec=max_gap_sec)
    results: List[Dict[str, Any]] = []

    for group in groups:
        start_sec = group[0]["time_sec"]
        end_sec = group[-1]["time_sec"]
        duration_sec = end_sec - start_sec
        if duration_sec < min_signal_frames / max(1.0, fps):
            continue

        pulse_count = estimate_pulse_count(group)
        classification = make_event_label(duration_sec, pulse_count)
        max_brightness = max(float(d["brightness"]) for d in group)
        mean_brightness = float(np.mean([d["brightness"] for d in group]))
        event = {
            "video": str(video_path),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "pulse_count": pulse_count,
            "max_brightness": round(max_brightness, 3),
            "mean_brightness": round(mean_brightness, 3),
            "classification": classification,
        }
        results.append(event)

        if save_thumbs:
            slide_dir = output_dir / "thumbs"
            slide_dir.mkdir(parents=True, exist_ok=True)
            # Export the frame nearest the center of the track.
            mid_idx = int(np.median([d["frame_idx"] for d in group]))
            # Re-open and seek to the relevant frame for a snapshot.
            snap = cv2.VideoCapture(str(video_path))
            snap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(mid_idx, total_frames - 1)))
            ok, frame = snap.read()
            snap.release()
            if ok:
                # Draw a bounding box around the strongest blob in the middle of the event.
                strongest = max(group, key=lambda d: d["brightness"])
                x = int(strongest["x"] * scale_x)
                y = int(strongest["y"] * scale_y)
                w = int(strongest["w"] * scale_x)
                h = int(strongest["h"] * scale_y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classification}_{int(strongest['time_sec'])}s"
                cv2.putText(frame, label, (max(10, x), max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out_path = slide_dir / f"{video_path.stem}_{int(start_sec)}_{classification}.png"
                cv2.imwrite(str(out_path), frame)

    return results


def append_lighting_sample(samples: List[Dict[str, Any]], video_path: Path, frame_idx: int,
                           fps: float, gray: np.ndarray) -> None:
    """Record robust brightness statistics from a frame-wide reference area."""
    mask = make_timestamp_mask(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).shape)
    values = gray[mask > 0]
    if values.size == 0:
        return
    samples.append({
        "night": extract_night(video_path),
        "video": str(video_path),
        "frame": int(frame_idx),
        "video_time_sec": round(frame_idx / fps, 3),
        "brightness_median": round(float(np.median(values)), 3),
        "brightness_p10": round(float(np.percentile(values, 10)), 3),
        "brightness_p90": round(float(np.percentile(values, 90)), 3),
    })


def _video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps > 0 and frames > 0 else 0.0


def write_lighting_analysis(samples: List[Dict[str, Any]], videos: List[Path], output_dir: Path,
                            sample_interval_sec: float) -> None:
    """Write per-frame brightness samples and conservative light transitions."""
    if not samples:
        print("No lighting samples were collected.")
        return

    video_order = {str(path): index for index, path in enumerate(sorted(videos, key=video_sort_key))}
    offsets: Dict[str, float] = {}
    elapsed = 0.0
    for path in sorted(videos, key=video_sort_key):
        offsets[str(path)] = elapsed
        elapsed += _video_duration(path)

    sample_df = pd.DataFrame(samples)
    sample_df["video_order"] = sample_df["video"].map(video_order)
    sample_df["night_elapsed_sec"] = sample_df["video"].map(offsets) + sample_df["video_time_sec"]
    sample_df.to_csv(output_dir / "lighting_samples.csv", index=False)

    transitions: List[Dict[str, Any]] = []
    for night, night_df in sample_df.groupby("night", dropna=False):
        ordered = night_df.sort_values("night_elapsed_sec").reset_index(drop=True)
        if len(ordered) < 5:
            continue
        signal = ordered["brightness_median"].rolling(3, center=True, min_periods=1).median()
        changes = signal.diff()
        robust_scale = float(np.median(np.abs(changes.dropna() - np.median(changes.dropna())))) if changes.notna().any() else 0.0
        threshold = max(8.0, 6.0 * robust_scale)
        down_index = int(changes.idxmin()) if changes.notna().any() else -1
        up_index = int(changes.idxmax()) if changes.notna().any() else -1

        row: Dict[str, Any] = {
            "night": str(night),
            "lights_off_detected": False,
            "lights_off_video": "",
            "lights_off_video_sec": np.nan,
            "lights_off_elapsed_sec": np.nan,
            "lights_off_brightness_change": np.nan,
            "lights_off_confidence": 0.0,
            "lights_on_detected": False,
            "lights_on_video": "",
            "lights_on_video_sec": np.nan,
            "lights_on_elapsed_sec": np.nan,
            "lights_on_brightness_change": np.nan,
            "lights_on_confidence": 0.0,
            "method": "median grayscale change across fixed frame reference area",
        }
        if down_index > 0 and float(changes.iloc[down_index]) <= -threshold:
            point = ordered.iloc[down_index]
            change = float(changes.iloc[down_index])
            row.update({
                "lights_off_detected": True,
                "lights_off_video": point["video"],
                "lights_off_video_sec": point["video_time_sec"],
                "lights_off_elapsed_sec": point["night_elapsed_sec"],
                "lights_off_brightness_change": change,
                "lights_off_confidence": round(min(1.0, abs(change) / 50.0), 3),
            })
        if up_index > 0 and float(changes.iloc[up_index]) >= threshold:
            point = ordered.iloc[up_index]
            change = float(changes.iloc[up_index])
            row.update({
                "lights_on_detected": True,
                "lights_on_video": point["video"],
                "lights_on_video_sec": point["video_time_sec"],
                "lights_on_elapsed_sec": point["night_elapsed_sec"],
                "lights_on_brightness_change": change,
                "lights_on_confidence": round(min(1.0, abs(change) / 50.0), 3),
            })
        transitions.append(row)

    pd.DataFrame(transitions).to_csv(output_dir / "lighting_transitions.csv", index=False)
    print(f"Wrote lighting samples to {output_dir / 'lighting_samples.csv'}")
    print(f"Wrote lighting transitions to {output_dir / 'lighting_transitions.csv'}")


def collect_lighting_samples(video_path: Path, sample_interval_sec: float) -> List[Dict[str, Any]]:
    """Sample a video sparsely for lighting analysis without contour detection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open video for lighting analysis: {video_path}", file=sys.stderr)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(fps * max(1.0, sample_interval_sec))))
    samples: List[Dict[str, Any]] = []
    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        append_lighting_sample(samples, video_path, frame_idx, fps, gray)
    cap.release()
    return samples


def scan_video_contours(video_path: Path, sample_rate_hz: float, black: int, min_area: float,
                        max_area: float, brightness_threshold: float,
                        start_sec: Optional[float], end_sec: Optional[float],
                        roi: Optional[Tuple[float, float, float, float]],
                        exclude_roi: Optional[Tuple[float, float, float, float]],
                        output_dir: Path, save_thumbs: bool,
                        thumb_time_gap_sec: float, thumb_position_gap_px: int,
                        on_detection: Optional[Callable[[Dict[str, Any]], None]] = None,
                        lighting_samples: Optional[List[Dict[str, Any]]] = None,
                        lighting_sample_sec: float = 15.0) -> List[Dict[str, Any]]:
    """Return independent bright contours from sampled frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_step = max(1, int(round(fps / sample_rate_hz)))
    first_frame = max(0, int((start_sec or 0.0) * fps))
    last_frame = None if end_sec is None else int(end_sec * fps)
    if first_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    detections: List[Dict[str, Any]] = []
    thumb_states: List[Tuple[float, int, int]] = []
    thumb_dir = output_dir / "thumbs"
    if save_thumbs:
        thumb_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = first_frame
    next_lighting_sample_sec = float(start_sec or 0.0)
    next_progress_frame = first_frame
    progress_step = max(1, int(fps * 30))
    while True:
        ok, frame = cap.read()
        if not ok or (last_frame is not None and frame_idx > last_frame):
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time_sec = frame_idx / fps
        if lighting_samples is not None and current_time_sec >= next_lighting_sample_sec:
            append_lighting_sample(lighting_samples, video_path, frame_idx, fps, gray)
            next_lighting_sample_sec += max(1.0, lighting_sample_sec)
        if brightness_threshold > 0 and float(cv2.mean(gray)[0]) > brightness_threshold:
            frame_idx += 1
            continue

        threshold_mask = cv2.inRange(gray, black, 255)
        threshold_mask = cv2.bitwise_and(threshold_mask, threshold_mask,
                                         mask=make_timestamp_mask(frame.shape))
        if roi is not None:
            roi_x0, roi_y0, roi_x1, roi_y1 = roi
            h, w = gray.shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            x0, y0 = int(w * roi_x0), int(h * roi_y0)
            x1, y1 = int(w * roi_x1), int(h * roi_y1)
            roi_mask[y0:y1, x0:x1] = 255
            threshold_mask = cv2.bitwise_and(threshold_mask, roi_mask)
        if exclude_roi is not None:
            exclude_x0, exclude_y0, exclude_x1, exclude_y1 = exclude_roi
            h, w = gray.shape[:2]
            x0, y0 = int(w * exclude_x0), int(h * exclude_y0)
            x1, y1 = int(w * exclude_x1), int(h * exclude_y1)
            threshold_mask[y0:y1, x0:x1] = 0

        contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            min_intensity, max_intensity, _, _ = cv2.minMaxLoc(gray, mask=contour_mask)
            mean_intensity = float(cv2.mean(gray, mask=contour_mask)[0])
            detections.append({
                "night": extract_night(video_path),
                "video": str(video_path),
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "x": centroid_x,
                "y": centroid_y,
                "y_flipped": gray.shape[0] - centroid_y,
                "area": round(area, 3),
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
                "mean_intensity": round(mean_intensity, 3),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
            })
            if on_detection is not None:
                on_detection(detections[-1])

            current_time = frame_idx / fps
            close_to_existing = any(
                abs(current_time - prior_time) <= thumb_time_gap_sec
                and abs(centroid_x - prior_x) <= thumb_position_gap_px
                and abs(centroid_y - prior_y) <= thumb_position_gap_px
                for prior_time, prior_x, prior_y in thumb_states
            )
            if save_thumbs and not close_to_existing:
                annotated = frame.copy()
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 165, 255), 3)
                label = f"{current_time:.1f}s x={centroid_x} y={centroid_y}"
                cv2.putText(annotated, label, (max(10, x), max(30, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                thumb_path = thumb_dir / f"{video_path.stem}_{frame_idx}_{centroid_x}_{centroid_y}.png"
                cv2.imwrite(str(thumb_path), annotated)
                thumb_states.append((current_time, centroid_x, centroid_y))
        frame_idx += 1
        if frame_idx >= next_progress_frame:
            print(f"Progress: {frame_idx / fps:.1f}s", flush=True)
            next_progress_frame += progress_step

    cap.release()
    return detections


def scan_directory(input_dir: Path, output_dir: Path, sample_rate_hz: float, black: int,
                   min_area: float, max_area: float, brightness_threshold: float,
                   start_sec: Optional[float] = None,
                   end_sec: Optional[float] = None,
                   roi: Optional[Tuple[float, float, float, float]] = None,
                   exclude_roi: Optional[Tuple[float, float, float, float]] = None,
                   save_thumbs: bool = False,
                   thumb_time_gap_sec: float = 5.0,
                   thumb_position_gap_px: int = 15,
                   lighting_sample_sec: float = 15.0,
                   lighting_only: bool = False) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_videos(input_dir)
    if not videos:
        print(f"No video files found under: {input_dir}", file=sys.stderr)
        return []

    all_results: List[Dict[str, Any]] = []
    lighting_samples: List[Dict[str, Any]] = []
    for video_path in videos:
        if lighting_only:
            print(f"Sampling lighting: {video_path}")
            lighting_samples.extend(collect_lighting_samples(video_path, lighting_sample_sec))
            continue
        print(f"Scanning: {video_path}")
        hits = scan_video_contours(
            video_path=video_path,
            sample_rate_hz=sample_rate_hz,
            black=black,
            min_area=min_area,
            max_area=max_area,
            brightness_threshold=brightness_threshold,
            start_sec=start_sec,
            end_sec=end_sec,
            roi=roi,
            exclude_roi=exclude_roi,
            output_dir=output_dir,
            save_thumbs=save_thumbs,
            thumb_time_gap_sec=thumb_time_gap_sec,
            thumb_position_gap_px=thumb_position_gap_px,
            lighting_samples=lighting_samples,
            lighting_sample_sec=lighting_sample_sec,
        )
        all_results.extend(hits)

    csv_path = output_dir / "bioluminescence_hits.csv"
    if not lighting_only:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(all_results[0].keys()) if all_results else ["video"],
            )
            writer.writeheader()
            writer.writerows(all_results)

    write_lighting_analysis(lighting_samples, videos, output_dir, lighting_sample_sec)

    if not lighting_only:
        print(f"\nWrote {len(all_results)} candidate events to {csv_path}")
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan video folders for bioluminescent events.")
    parser.add_argument("input_path", nargs="?", default=None, help="Folder containing videos or a mounted SMB path, e.g. /Volumes/oakley_lab/...")
    parser.add_argument("--video", help="Optional path to a single video file to inspect; this is useful for checking positive-control clips like output4.")
    parser.add_argument(
        "--output",
        help="Output folder for CSV and optional thumbnails (default: Oakley NAS).",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=2.5, help="Sample frames at this rate.")
    parser.add_argument("--black", type=int, default=110, help="Pixel intensity threshold for candidate blobs.")
    parser.add_argument("--min-area", type=float, default=1.5, help="Minimum contour area in pixels.")
    parser.add_argument("--max-area", type=float, default=1000.0, help="Maximum contour area in pixels.")
    parser.add_argument("--brightness-threshold", type=float, default=200.0, help="Skip frames whose average grayscale brightness exceeds this value; 0 disables skipping.")
    parser.add_argument(
        "--roi",
        help="Optional normalized detection region x0,y0,x1,y1; e.g. 0.55,0.80,0.75,1.0",
    )
    parser.add_argument(
        "--exclude-roi",
        help="Optional normalized region to ignore x0,y0,x1,y1; e.g. 0,0,0.12,0.25",
    )
    parser.add_argument("--start-sec", type=float, help="Only scan from this video timestamp in seconds.")
    parser.add_argument("--end-sec", type=float, help="Only scan through this video timestamp in seconds.")
    parser.add_argument("--save-thumbs", action="store_true", help="Save representative annotated thumbnails.")
    parser.add_argument("--thumb-time-gap-sec", type=float, default=5.0, help="Save another thumbnail after this time gap.")
    parser.add_argument("--thumb-position-gap-px", type=int, default=15, help="Save another thumbnail when x or y differs by this many pixels.")
    parser.add_argument("--lighting-sample-sec", type=float, default=15.0, help="Sample frame-wide brightness every N seconds for lights-off/on detection.")
    parser.add_argument("--lighting-only", action="store_true", help="Skip blob detection and only sample lighting transitions.")
    return parser.parse_args()


def parse_roi(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("ROI must be four comma-separated numbers: x0,y0,x1,y1") from exc
    if len(parts) != 4 or not all(0.0 <= part <= 1.0 for part in parts):
        raise ValueError("ROI coordinates must be four values between 0 and 1: x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have x1 > x0 and y1 > y0")
    return parts


def main() -> int:
    args = parse_args()

    try:
        roi = parse_roi(args.roi)
        exclude_roi = parse_roi(args.exclude_roi)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.black < 0 or args.black > 255:
        print("ERROR: black threshold must be between 0 and 255", file=sys.stderr)
        return 2
    if args.start_sec is not None and args.start_sec < 0:
        print("ERROR: start timestamp must be non-negative", file=sys.stderr)
        return 2
    if args.end_sec is not None and args.start_sec is not None and args.end_sec <= args.start_sec:
        print("ERROR: end timestamp must be greater than start timestamp", file=sys.stderr)
        return 2

    if args.video is None and args.input_path is None:
        print("ERROR: provide either an input_path or --video", file=sys.stderr)
        return 2

    if args.input_path is not None:
        try:
            input_path = Path(normalize_input_path(args.input_path))
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if not input_path.exists():
            print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
            return 2
    else:
        input_path = None

    try:
        output_dir = (
            Path(args.output).expanduser() if args.output else default_nas_output_dir()
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(normalize_input_path(args.video))
        if not video_path.exists():
            print(f"ERROR: Video file does not exist: {video_path}", file=sys.stderr)
            return 2
        print(f"Scanning single video: {video_path}")
        csv_path = output_dir / f"{video_path.stem}_hits.csv"
        count = 0
        lighting_samples: List[Dict[str, Any]] = []
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=DETECTION_FIELDS)
            writer.writeheader()
            rows_since_flush = 0

            def write_detection(row: Dict[str, Any]) -> None:
                nonlocal count, rows_since_flush
                writer.writerow(row)
                count += 1
                rows_since_flush += 1
                if rows_since_flush >= 100:
                    fh.flush()
                    rows_since_flush = 0

            scan_video_contours(
                video_path=video_path,
                sample_rate_hz=args.sample_rate_hz,
                black=args.black,
                min_area=args.min_area,
                max_area=args.max_area,
                brightness_threshold=args.brightness_threshold,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                roi=roi,
                exclude_roi=exclude_roi,
                output_dir=output_dir,
                save_thumbs=args.save_thumbs,
                thumb_time_gap_sec=args.thumb_time_gap_sec,
                thumb_position_gap_px=args.thumb_position_gap_px,
                on_detection=write_detection,
                lighting_samples=lighting_samples,
                lighting_sample_sec=args.lighting_sample_sec,
            )
            fh.flush()
        write_lighting_analysis(lighting_samples, [video_path], output_dir, args.lighting_sample_sec)
        print(f"Wrote {count} hits to {csv_path}")
        return 0

    if not input_path.is_dir():
        print(f"ERROR: Input path is not a directory: {input_path}", file=sys.stderr)
        return 2

    print(f"Scanning directory: {input_path}")
    scan_directory(
        input_dir=input_path,
        output_dir=output_dir,
        sample_rate_hz=args.sample_rate_hz,
        black=args.black,
        min_area=args.min_area,
        max_area=args.max_area,
        brightness_threshold=args.brightness_threshold,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        roi=roi,
        exclude_roi=exclude_roi,
        save_thumbs=args.save_thumbs,
        thumb_time_gap_sec=args.thumb_time_gap_sec,
        thumb_position_gap_px=args.thumb_position_gap_px,
        lighting_sample_sec=args.lighting_sample_sec,
        lighting_only=args.lighting_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

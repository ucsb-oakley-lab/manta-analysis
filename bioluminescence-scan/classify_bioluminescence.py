#!/usr/bin/env python3
"""Classify scan detections as likely sand reflection or bioluminescence.

The classifier preserves every input row and adds classification features.  It
uses the first frame of the earliest timestamped video for each night to build
a sand reference, then combines that spatial reference with brightness,
morphology, and short-term spatial persistence.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from moon_state import add_moon_state_metadata


TIMESTAMP_RE = re.compile(r"20\d{6}T\d{6}Z")


def timestamp_key(value: str) -> tuple[int, str]:
    """Sort timestamped files before legacy files."""
    match = TIMESTAMP_RE.search(Path(str(value)).name)
    return (0, match.group(0)) if match else (1, Path(str(value)).name)


def read_first_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def segment_sand(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask for the bright, connected sand region.

    The camera view has sand along the lower edge. Otsu thresholding is
    restricted to the lower half, then only components connected to the lower
    edge are retained. This avoids treating bright timestamp text or equipment
    as sand.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    lower = gray[int(height * 0.50) :]
    threshold, _ = cv2.threshold(lower, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(threshold), float(np.percentile(lower, 55)))
    candidate = np.zeros_like(gray, dtype=np.uint8)
    candidate[int(height * 0.50) :] = (lower >= threshold).astype(np.uint8) * 255
    kernel = np.ones((9, 9), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    mask = np.zeros_like(candidate)
    bottom_row = height - 1
    components = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= 500
        and labels[bottom_row, :].tolist().count(label) > 0
    ]
    if components:
        mask[np.isin(labels, components)] = 255
    return mask


def save_sand_regions(mask: np.ndarray, night: str, output_dir: Path) -> list[dict[str, object]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: list[dict[str, object]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 500:
            continue
        regions.append(
            {
                "night": night,
                "label": label,
                "area_pixels": int(area),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_width": int(width),
                "bbox_height": int(height),
                "centroid_x": float(centroids[label][0]),
                "centroid_y": float(centroids[label][1]),
            }
        )
    cv2.imwrite(str(output_dir / f"sand_mask_{night}.png"), mask)
    return regions


def build_reference(
    df: pd.DataFrame, output_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]]]:
    masks: dict[str, np.ndarray] = {}
    reference_grays: dict[str, np.ndarray] = {}
    regions: list[dict[str, object]] = []
    timestamped = df[df["video"].map(lambda value: bool(TIMESTAMP_RE.search(Path(str(value)).name)))].copy()
    for night, night_df in timestamped.groupby(timestamped["night"].astype(str)):
        video = sorted(night_df["video"].dropna().unique(), key=timestamp_key)[0]
        frame = read_first_frame(Path(video))
        if frame is None:
            print(f"Warning: could not read reference frame for {night}: {video}")
            continue
        mask = segment_sand(frame)
        masks[night] = mask
        reference_grays[night] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        regions.extend(save_sand_regions(mask, night, output_dir))
    return masks, reference_grays, regions


def add_track_features(df: pd.DataFrame, track_size: int) -> pd.DataFrame:
    result = df.copy()
    result["bbox_area"] = result["bbox_width"] * result["bbox_height"]
    result["compactness"] = result["area"] / result["bbox_area"].replace(0, np.nan)
    result["spatial_track_x"] = np.rint(result["x"] / track_size).astype("Int64")
    result["spatial_track_y"] = np.rint(result["y"] / track_size).astype("Int64")
    group_columns = ["video", "spatial_track_x", "spatial_track_y"]
    grouped = result.groupby(group_columns, dropna=False)
    result["track_count"] = grouped["frame"].transform("size")
    result["track_first_sec"] = grouped["time_sec"].transform("min")
    result["track_last_sec"] = grouped["time_sec"].transform("max")
    result["track_span_sec"] = result["track_last_sec"] - result["track_first_sec"]
    return result


def classify(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    reference_grays: dict[str, np.ndarray],
    min_track_span: float,
) -> pd.DataFrame:
    result = add_track_features(df, track_size=12)
    result["in_sand_reference"] = False
    result["reference_brightness"] = np.nan

    for index, row in result.iterrows():
        night = str(row["night"])
        mask = masks.get(night)
        gray = reference_grays.get(night)
        if mask is None or gray is None:
            continue
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            continue
        radius = 5
        y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        patch = mask[y0:y1, x0:x1]
        result.at[index, "in_sand_reference"] = bool(np.mean(patch > 0) >= 0.35)
        result.at[index, "reference_brightness"] = float(np.mean(gray[y0:y1, x0:x1]))

    result["brightness_lift"] = result["mean_intensity"] - result["reference_brightness"]
    track_columns = ["video", "spatial_track_x", "spatial_track_y"]
    track_groups = result.groupby(track_columns, dropna=False)
    result["track_bright_fraction"] = track_groups["mean_intensity"].transform(
        lambda values: float(((values >= 210) & (result.loc[values.index, "max_intensity"] >= 250)).mean())
    )
    result["track_peak_mean_intensity"] = track_groups["mean_intensity"].transform("max")
    result["reflection_score"] = 0
    result.loc[result["in_sand_reference"], "reflection_score"] += 1
    result.loc[result["track_span_sec"] >= min_track_span, "reflection_score"] += 1
    result.loc[result["mean_intensity"] < 190, "reflection_score"] += 1
    result.loc[result["max_intensity"] < 245, "reflection_score"] += 1
    result.loc[result["compactness"] < 0.35, "reflection_score"] += 1

    result["classification"] = "uncertain"
    reflection = result["reflection_score"] >= 3
    bright_bottom = (
        (result["mean_intensity"] >= 210) & (result["max_intensity"] >= 250)
    ) | (result["track_bright_fraction"] >= 0.25)
    result.loc[reflection & ~bright_bottom, "classification"] = "reflection_suspect"
    result.loc[bright_bottom & (result["reflection_score"] <= 2), "classification"] = "bottom_bioluminescence"
    result["classification_reason"] = np.select(
        [result["classification"].eq("reflection_suspect"), result["classification"].eq("bottom_bioluminescence")],
        ["sand reference + persistence/brightness/morphology", "bright, saturated, compact pulse despite spatial persistence"],
        default="mixed or insufficient evidence",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw bioluminescence detections CSV")
    parser.add_argument("--output", type=Path, required=True, help="Classified detections CSV")
    parser.add_argument("--reference-output", type=Path, default=None, help="Directory for sand masks and region CSV")
    parser.add_argument("--simulator-log", type=Path, default=None, help="CSV log anchoring simulator time to actual wall-clock time")
    parser.add_argument("--min-track-span", type=float, default=30.0, help="Seconds for a persistent spatial track")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"night", "video", "frame", "time_sec", "x", "y", "area", "mean_intensity", "max_intensity", "bbox_width", "bbox_height"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
    if args.simulator_log is not None:
        if not args.simulator_log.exists():
            raise SystemExit(f"Simulator log does not exist: {args.simulator_log}")
        df = add_moon_state_metadata(df, args.simulator_log)
    reference_output = args.reference_output or args.output.parent / "sand_references"
    reference_output.mkdir(parents=True, exist_ok=True)
    masks, reference_grays, regions = build_reference(df, reference_output)
    result = classify(df, masks, reference_grays, args.min_track_span)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    pd.DataFrame(regions).to_csv(reference_output / "sand_regions.csv", index=False)
    print(f"Wrote {len(result)} classified detections to {args.output}")
    print(f"Wrote {len(regions)} sand reference regions to {reference_output / 'sand_regions.csv'}")
    print(result["classification"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

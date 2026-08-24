#!/usr/bin/env python3
"""Quickly screen timestamped videos for possible bioluminescent flashes.

Video filenames are assumed to contain a UTC date and time. Measurements are
written to one CSV and plotted against local clock time (America/Panama by
default). The default scans every frame, but reduces large frames before
calculating brightness metrics to keep the first-pass screen fast.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg"}
TIMESTAMP_PATTERNS = (
    # 20240826_031425, 2024-08-26T03-14-25, 2024.08.26 031425, etc.
    re.compile(
        r"(?<!\d)(?P<year>20\d{2})[-_.]?(?P<month>\d{2})[-_.]?"
        r"(?P<day>\d{2})[T _.-]+(?P<hour>\d{2})[-_.:]?"
        r"(?P<minute>\d{2})[-_.:]?(?P<second>\d{2})(?!\d)"
    ),
    # 26Aug2024_031425
    re.compile(
        r"(?<!\d)(?P<day>\d{2})(?P<month_name>[A-Za-z]{3})"
        r"(?P<year>20\d{2})[T _.-]+(?P<hour>\d{2})[-_.:]?"
        r"(?P<minute>\d{2})[-_.:]?(?P<second>\d{2})(?!\d)"
    ),
)


@dataclass(frozen=True)
class Measurement:
    video: str
    frame: int
    seconds: float
    time_utc: datetime
    time_local: datetime
    mean: float
    top_mean: float
    percentile: float
    maximum: int
    bright_fraction: float


def parse_utc_start(filename: str) -> datetime | None:
    """Extract a timezone-aware UTC start time from a video filename."""
    stem = Path(filename).stem
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        fields = match.groupdict()
        try:
            if fields.get("month_name"):
                month = datetime.strptime(fields["month_name"], "%b").month
            else:
                month = int(fields["month"])
            return datetime(
                int(fields["year"]),
                month,
                int(fields["day"]),
                int(fields["hour"]),
                int(fields["minute"]),
                int(fields["second"]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
    return None


def find_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTENSIONS else []
    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def resize_for_screening(gray: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or gray.shape[1] <= max_width:
        return gray
    scale = max_width / gray.shape[1]
    size = (max_width, max(1, round(gray.shape[0] * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def brightness_metrics(
    gray: np.ndarray, top_fraction: float, percentile: float, bright_threshold: int
) -> tuple[float, float, float, int, float]:
    flat = gray.reshape(-1)
    top_count = max(1, int(round(flat.size * top_fraction)))
    split = flat.size - top_count
    top_mean = float(np.partition(flat, split)[split:].mean())
    return (
        float(flat.mean()),
        top_mean,
        float(np.percentile(flat, percentile)),
        int(flat.max()),
        float(np.count_nonzero(flat >= bright_threshold) / flat.size),
    )


def scan_video(
    path: Path,
    start_utc: datetime,
    local_zone: ZoneInfo,
    frame_step: int,
    max_width: int,
    top_fraction: float,
    percentile: float,
    bright_threshold: int,
    region: str = "full",
) -> list[Measurement]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("video has no usable frame rate")

    rows: list[Measurement] = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if region == "left":
                    gray = gray[:, : gray.shape[1] // 2]
                elif region == "right":
                    gray = gray[:, gray.shape[1] // 2 :]
                gray = resize_for_screening(gray, max_width)
                mean, top_mean, pct, maximum, bright_fraction = brightness_metrics(
                    gray, top_fraction, percentile, bright_threshold
                )
                seconds = frame_number / fps
                time_utc = start_utc + timedelta(seconds=seconds)
                rows.append(
                    Measurement(
                        path.name,
                        frame_number,
                        seconds,
                        time_utc,
                        time_utc.astimezone(local_zone),
                        mean,
                        top_mean,
                        pct,
                        maximum,
                        bright_fraction,
                    )
                )
            frame_number += 1
    finally:
        capture.release()
    return rows


def write_csv(rows: list[Measurement], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "video",
                "frame",
                "seconds_from_video_start",
                "time_utc",
                "time_panama",
                "mean_brightness",
                "top_fraction_mean",
                "brightness_percentile",
                "max_brightness",
                "fraction_above_threshold",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.video,
                    row.frame,
                    f"{row.seconds:.6f}",
                    row.time_utc.isoformat(timespec="microseconds"),
                    row.time_local.isoformat(timespec="microseconds"),
                    f"{row.mean:.6f}",
                    f"{row.top_mean:.6f}",
                    f"{row.percentile:.6f}",
                    row.maximum,
                    f"{row.bright_fraction:.9f}",
                ]
            )


def write_plot(rows: list[Measurement], path: Path, timezone_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [row.time_local for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(times, [row.mean for row in rows], linewidth=0.7, color="0.25")
    axes[0].set_ylabel("Mean brightness (0–255)")
    axes[0].grid(alpha=0.2)
    axes[1].plot(times, [row.top_mean for row in rows], linewidth=0.7, color="#0072B2")
    axes[1].set_ylabel("Brightest-pixel mean (0–255)")
    axes[1].set_xlabel(f"Clock time ({timezone_name})")
    axes[1].grid(alpha=0.2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("Video brightness screening")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="video file or directory of videos")
    parser.add_argument("--output-csv", type=Path, default=Path("brightness_over_time.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("brightness_over_time.png"))
    parser.add_argument("--timezone", default="America/Panama", help="output/plot IANA timezone")
    parser.add_argument("--recursive", action="store_true", help="search subdirectories")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="analyze every Nth frame (default: every frame; use 2 or 3 for a faster preview)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=640,
        help="downscale wider frames before measurement; 0 keeps full resolution (default: 640)",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.001,
        help="fraction of brightest pixels averaged for the spot-sensitive metric (default: 0.001)",
    )
    parser.add_argument(
        "--percentile", type=float, default=99.9, help="brightness percentile to record (default: 99.9)"
    )
    parser.add_argument(
        "--bright-threshold",
        type=int,
        default=200,
        help="pixel value used for fraction_above_threshold (default: 200)",
    )
    parser.add_argument(
        "--region",
        choices=("full", "left", "right"),
        default="full",
        help="measure the full frame or one half of a side-by-side video (default: full)",
    )
    parser.add_argument(
        "--skip-unparseable",
        action="store_true",
        help="skip videos whose filenames do not contain a recognizable UTC timestamp",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        print(f"error: input does not exist: {args.input}", file=sys.stderr)
        return 2
    if args.frame_step < 1:
        print("error: --frame-step must be at least 1", file=sys.stderr)
        return 2
    if not 0 < args.top_fraction <= 1:
        print("error: --top-fraction must be in (0, 1]", file=sys.stderr)
        return 2
    if not 0 <= args.percentile <= 100:
        print("error: --percentile must be in [0, 100]", file=sys.stderr)
        return 2
    if not 0 <= args.bright_threshold <= 255:
        print("error: --bright-threshold must be in [0, 255]", file=sys.stderr)
        return 2
    try:
        local_zone = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError:
        print(f"error: unknown timezone: {args.timezone}", file=sys.stderr)
        return 2

    videos = find_videos(args.input, args.recursive)
    if not videos:
        print(f"error: no supported videos found in {args.input}", file=sys.stderr)
        return 1

    dated_videos: list[tuple[datetime, Path]] = []
    for video in videos:
        start = parse_utc_start(video.name)
        if start is None:
            message = f"filename has no recognizable UTC timestamp: {video.name}"
            if args.skip_unparseable:
                print(f"warning: skipping; {message}", file=sys.stderr)
                continue
            print(f"error: {message}", file=sys.stderr)
            print("Use --skip-unparseable or rename it with YYYYMMDD_HHMMSS.", file=sys.stderr)
            return 2
        dated_videos.append((start, video))
    dated_videos.sort()

    all_rows: list[Measurement] = []
    for index, (start, video) in enumerate(dated_videos, start=1):
        print(f"[{index}/{len(dated_videos)}] {video.name}", flush=True)
        try:
            all_rows.extend(
                scan_video(
                    video,
                    start,
                    local_zone,
                    args.frame_step,
                    args.max_width,
                    args.top_fraction,
                    args.percentile,
                    args.bright_threshold,
                    args.region,
                )
            )
        except RuntimeError as error:
            print(f"warning: skipping {video.name}: {error}", file=sys.stderr)

    if not all_rows:
        print("error: no frames were measured", file=sys.stderr)
        return 1
    all_rows.sort(key=lambda row: row.time_utc)
    write_csv(all_rows, args.output_csv)
    write_plot(all_rows, args.output_plot, args.timezone)
    print(f"Wrote {len(all_rows):,} measurements to {args.output_csv}")
    print(f"Wrote plot to {args.output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

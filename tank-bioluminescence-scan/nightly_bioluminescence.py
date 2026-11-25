#!/usr/bin/env python3
"""Helpers for storing and aggregating per-night bioluminescence CSVs."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Any, TextIO


NIGHT_RE = re.compile(r"20\d{6}")
NAS_OUTPUT_SUBDIR = Path(
    "analysis/manta-analysis/tank-bioluminescence-scan/output"
)


def nas_mountpoint() -> Path:
    for candidate in (Path.home() / "labdata", Path("/Volumes/oakley_lab")):
        if candidate.is_mount():
            return candidate
    raise FileNotFoundError(
        "Oakley NAS is not mounted. Mount it at ~/labdata (Linux) "
        "or /Volumes/oakley_lab (macOS)."
    )


def default_nas_output_dir() -> Path:
    return nas_mountpoint() / NAS_OUTPUT_SUBDIR


def night_csv_path(output_dir: Path, night: str, *, classified: bool = False) -> Path:
    if not NIGHT_RE.fullmatch(str(night)):
        raise ValueError(f"Invalid night identifier: {night!r}")
    suffix = "_hits_classified.csv" if classified else "_hits.csv"
    return output_dir / f"{night}{suffix}"


def list_night_csvs(output_dir: Path, *, classified: bool = False) -> list[Path]:
    suffix = "_hits_classified.csv" if classified else "_hits.csv"
    pattern = re.compile(rf"(20\d{{6}}){re.escape(suffix)}")
    return sorted(
        path for path in output_dir.glob(f"*{suffix}") if pattern.fullmatch(path.name)
    )


def _night_from_video(value: str) -> str:
    match = NIGHT_RE.search(value.replace("\\", "/"))
    if match is None:
        raise ValueError(f"Cannot determine night from video path: {value!r}")
    return match.group(0)


def split_master_csv_by_night(
    master_csv: Path, output_dir: Path, *, classified: bool = False
) -> list[Path]:
    """Create missing dated CSVs from a combined CSV without changing its rows."""
    if not master_csv.is_file():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_hits_classified.csv" if classified else "_hits.csv"
    existing_nights = {
        path.name[: -len(suffix)]
        for path in list_night_csvs(output_dir, classified=classified)
    }

    with tempfile.TemporaryDirectory(prefix=".nightly_split_", dir=output_dir) as temp_dir:
        staged_dir = Path(temp_dir)
        writers: dict[str, tuple[TextIO, Any]] = {}
        staged_paths: dict[str, Path] = {}
        try:
            with master_csv.open(newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                try:
                    header = next(reader)
                except StopIteration:
                    return []
                night_index = header.index("night") if "night" in header else None
                video_index = header.index("video") if "video" in header else None
                if night_index is None and video_index is None:
                    raise ValueError(f"{master_csv} must contain a 'night' or 'video' column")
                output_header = header if night_index is not None else [*header, "night"]

                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(f"Malformed CSV row in {master_csv}")
                    night = row[night_index].strip() if night_index is not None else ""
                    if not NIGHT_RE.fullmatch(night):
                        if video_index is None:
                            raise ValueError(f"Invalid night value in {master_csv}: {night!r}")
                        night = _night_from_video(row[video_index])
                    if night in existing_nights:
                        continue
                    if night not in writers:
                        staged_path = staged_dir / f"{night}{suffix}"
                        stream = staged_path.open("w", newline="", encoding="utf-8")
                        writer = csv.writer(stream)
                        writer.writerow(output_header)
                        writers[night] = (stream, writer)
                        staged_paths[night] = staged_path
                    if night_index is None:
                        output_row = [*row, night]
                    else:
                        row[night_index] = night
                        output_row = row
                    writers[night][1].writerow(output_row)
        finally:
            for stream, _ in writers.values():
                stream.close()

        created: list[Path] = []
        for night, staged_path in staged_paths.items():
            destination = night_csv_path(output_dir, night, classified=classified)
            if not destination.exists():
                os.replace(staged_path, destination)
                created.append(destination)
        return created


def write_scanned_night(
    staged_csv: Path, output_dir: Path, night: str
) -> tuple[Path, int]:
    """Atomically save a scan as the night raw CSV and invalidate its old classification."""
    raw_path = night_csv_path(output_dir, night)
    classified_path = night_csv_path(output_dir, night, classified=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    row_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{night}_",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            writer = csv.writer(target)
            with staged_csv.open(newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"Scan output is empty: {staged_csv}") from exc
                if "night" in header:
                    night_index = header.index("night")
                    writer.writerow(header)
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"Malformed CSV row in {staged_csv}")
                        row[night_index] = night
                        writer.writerow(row)
                        row_count += 1
                else:
                    writer.writerow([*header, "night"])
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"Malformed CSV row in {staged_csv}")
                        writer.writerow([*row, night])
                        row_count += 1
        os.replace(temp_path, raw_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    classified_path.unlink(missing_ok=True)
    return raw_path, row_count


def classification_worklist(
    output_dir: Path, reclassify_nights: Collection[str] = ()
) -> list[tuple[str, Path, Path]]:
    worklist = []
    for raw_path in list_night_csvs(output_dir):
        night = raw_path.name[: -len("_hits.csv")]
        classified_path = night_csv_path(output_dir, night, classified=True)
        if not classified_path.is_file() or night in reclassify_nights:
            worklist.append((night, raw_path, classified_path))
    return worklist


def rebuild_master_csv(
    output_dir: Path, master_csv: Path, *, classified: bool = False
) -> int:
    """Regenerate a combined CSV by streaming dated files in night order."""
    inputs = list_night_csvs(output_dir, classified=classified)
    if not inputs:
        return 0

    master_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    row_count = 0
    header: list[str] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=master_csv.parent,
            prefix=f".{master_csv.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            writer = csv.writer(target)
            for input_csv in inputs:
                with input_csv.open(newline="", encoding="utf-8") as source:
                    reader = csv.reader(source)
                    try:
                        current_header = next(reader)
                    except StopIteration:
                        continue
                    if header is None:
                        header = current_header
                        writer.writerow(header)
                    elif current_header != header:
                        raise ValueError(f"CSV columns differ in {input_csv}")
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"Malformed CSV row in {input_csv}")
                        writer.writerow(row)
                        row_count += 1
        if header is None:
            temp_path.unlink(missing_ok=True)
            temp_path = None
            return 0
        os.replace(temp_path, master_csv)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return row_count


def rebuild_candidate_csv(
    output_dir: Path,
    candidate_csv: Path,
    *,
    classifications: Collection[str] = ("bottom_bioluminescence", "uncertain"),
) -> int:
    """Write only prospective detections from the per-night classified CSVs."""
    inputs = list_night_csvs(output_dir, classified=True)
    if not inputs:
        raise FileNotFoundError(f"No classified nightly CSVs found in {output_dir}")

    candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    row_count = 0
    header: list[str] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=candidate_csv.parent,
            prefix=f".{candidate_csv.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            writer = csv.writer(target, lineterminator="\n")
            for input_csv in inputs:
                with input_csv.open(newline="", encoding="utf-8") as source:
                    reader = csv.reader(source)
                    try:
                        current_header = next(reader)
                    except StopIteration:
                        continue
                    if "classification" not in current_header:
                        raise ValueError(
                            f"{input_csv} must contain a 'classification' column"
                        )
                    if header is None:
                        header = current_header
                        writer.writerow(header)
                    elif current_header != header:
                        raise ValueError(f"CSV columns differ in {input_csv}")
                    classification_index = current_header.index("classification")
                    for row in reader:
                        if len(row) != len(current_header):
                            raise ValueError(f"Malformed CSV row in {input_csv}")
                        if row[classification_index] in classifications:
                            writer.writerow(row)
                            row_count += 1
        if header is None:
            raise ValueError(f"No CSV headers found in classified files in {output_dir}")
        os.replace(temp_path, candidate_csv)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return row_count

#!/usr/bin/env python3
"""Infer the moon-simulator state at each detection.

The simulator is stationary at the configured latitude/longitude, but its
internal astronomical clock advances continuously in real time.  A manual log
entry anchors that simulator clock to an actual wall-clock time.  Timestamped
video filenames then provide the actual time for each detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astral import Observer
from astral import moon, sun


TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{6}Z)")
SYNODIC_MONTH_DAYS = 29.530588853


@dataclass(frozen=True)
class SimulatorAnchor:
    actual_check_time: datetime
    simulator_time_at_check: datetime
    latitude: float
    longitude: float
    simulator_timezone: str
    runs_continuously: bool

    @property
    def observer(self) -> Observer:
        return Observer(latitude=self.latitude, longitude=self.longitude)


def _parse_datetime(value: object, *, default_timezone: str | None = None) -> datetime:
    parsed = pd.to_datetime(str(value))
    if isinstance(parsed, pd.Timestamp):
        result = parsed.to_pydatetime()
    else:
        result = parsed
    if result.tzinfo is None:
        if default_timezone is None:
            return result
        result = result.replace(tzinfo=ZoneInfo(default_timezone))
    return result


def load_simulator_anchor(path: Path) -> SimulatorAnchor:
    """Load the first row of a simulator log CSV."""
    log = pd.read_csv(path)
    if log.empty:
        raise ValueError(f"Simulator log is empty: {path}")
    row = log.iloc[0]
    simulator_timezone = str(row.get("simulator_timezone", "America/Panama"))
    actual_check = _parse_datetime(row["actual_check_time"])
    simulator_time = _parse_datetime(
        row["simulator_time_at_check"], default_timezone=simulator_timezone
    )
    return SimulatorAnchor(
        actual_check_time=actual_check,
        simulator_time_at_check=simulator_time.replace(tzinfo=None),
        latitude=float(row["simulator_latitude"]),
        longitude=float(row["simulator_longitude"]),
        simulator_timezone=simulator_timezone,
        runs_continuously=str(row.get("simulator_runs_continuously", "true")).lower() in {"true", "1", "yes"},
    )


def video_timestamp(value: object) -> datetime | None:
    """Return the UTC start timestamp embedded in a video filename."""
    match = TIMESTAMP_RE.search(Path(str(value)).name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def moon_illumination_fraction(phase_age_days: float) -> float:
    """Approximate illuminated fraction from Astral's lunar age in days."""
    angle = 2.0 * np.pi * (phase_age_days % SYNODIC_MONTH_DAYS) / SYNODIC_MONTH_DAYS
    return float(0.5 * (1.0 - np.cos(angle)))


def continuous_moon_phase_age(simulator_dt: datetime) -> float:
    """Estimate lunar phase age continuously rather than once per calendar day."""
    new_moon_epoch = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    utc_dt = simulator_dt.astimezone(timezone.utc)
    elapsed_days = (utc_dt - new_moon_epoch).total_seconds() / 86400.0
    return float(elapsed_days % SYNODIC_MONTH_DAYS)


def add_moon_state_metadata(df: pd.DataFrame, simulator_log: Path) -> pd.DataFrame:
    """Add inferred simulator time and astronomical state to each detection."""
    anchor = load_simulator_anchor(simulator_log)
    result = df.copy()
    result["actual_detection_time_utc"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["inferred_simulator_time"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["moon_state_inference_status"] = pd.Series(
        "missing_video_timestamp", index=result.index, dtype="string"
    )
    result["simulator_latitude"] = np.nan
    result["simulator_longitude"] = np.nan
    result["sun_altitude_deg"] = np.nan
    result["moon_altitude_deg"] = np.nan
    result["moon_azimuth_deg"] = np.nan
    result["moon_phase_age_days"] = np.nan
    result["moon_illumination_fraction"] = np.nan
    result["moon_above_horizon"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["simulator_latitude"] = anchor.latitude
    result["simulator_longitude"] = anchor.longitude

    for index, row in result.iterrows():
        start_utc = video_timestamp(row.get("video", ""))
        if start_utc is None:
            continue
        try:
            offset = timedelta(seconds=float(row.get("time_sec", 0.0)))
        except (TypeError, ValueError):
            result.at[index, "moon_state_inference_status"] = "invalid_detection_time"
            continue
        actual_utc = start_utc + offset
        elapsed = actual_utc - anchor.actual_check_time.astimezone(timezone.utc)
        simulator_time = anchor.simulator_time_at_check + elapsed
        simulator_dt = simulator_time.replace(tzinfo=ZoneInfo(anchor.simulator_timezone))
        astronomy_time = simulator_dt.astimezone(timezone.utc)
        phase_age = continuous_moon_phase_age(simulator_dt)
        result.at[index, "actual_detection_time_utc"] = actual_utc.isoformat()
        result.at[index, "inferred_simulator_time"] = simulator_time.isoformat(sep=" ")
        result.at[index, "moon_state_inference_status"] = "inferred_from_timestamp_and_anchor"
        result.at[index, "sun_altitude_deg"] = float(sun.elevation(anchor.observer, astronomy_time))
        result.at[index, "moon_altitude_deg"] = float(moon.elevation(anchor.observer, astronomy_time))
        result.at[index, "moon_azimuth_deg"] = float(moon.azimuth(anchor.observer, astronomy_time))
        result.at[index, "moon_phase_age_days"] = phase_age
        result.at[index, "moon_illumination_fraction"] = moon_illumination_fraction(phase_age)
        result.at[index, "moon_above_horizon"] = bool(result.at[index, "moon_altitude_deg"] >= 0.0)

    return result

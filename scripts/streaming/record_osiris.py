#!/usr/bin/env python3
"""Manually record an RTSP camera to Osiris's mounted NAS (FFmpeg 4.4)."""

import argparse
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from camIP12h import log, stop_ffmpeg


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return seconds


def end_time(value):
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use an ISO date/time with UTC offset") from error
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError("include a UTC offset, such as -08:00")
    return result.timestamp()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    stop = parser.add_mutually_exclusive_group(required=True)
    stop.add_argument("--seconds", type=positive_seconds, help="recording session duration")
    stop.add_argument("--until", type=end_time, help="fixed stop time, e.g. 2026-11-16T06:00:00-08:00")
    parser.add_argument("--segment-seconds", type=positive_seconds, default=1800)
    parser.add_argument("--camera-url", default="rtsp://10.1.4.207:554/h264")
    parser.add_argument("--mount-point", type=Path, default=Path.home() / "labdata")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.access(args.ffmpeg, os.X_OK):
        raise SystemExit(f"FFmpeg is missing or not executable: {args.ffmpeg}")

    duration = args.seconds if args.seconds is not None else args.until - time.time()
    if duration <= 0:
        log("Stop time has already passed; no recording started.")
        return
    deadline = time.monotonic() + duration

    def remaining():
        left = deadline - time.monotonic()
        if args.until is not None:
            left = min(left, args.until - time.time())
        return max(0, left)

    # Resolve the home-directory alias used on Osiris before checking the mount.
    mount_point = args.mount_point.expanduser().resolve()
    recording_dir = mount_point / "aquarium" / "2026Pan" / datetime.now().strftime("%Y%m%d")
    run_id = uuid4().hex[:8]
    attempt = 0

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    log(f"Recording directory: {recording_dir}")
    log(f"Session budget: {duration:.0f} seconds; run ID: {run_id}")
    try:
        while remaining() > 0:
            if not os.path.ismount(mount_point):
                log(f"NAS is not mounted at {mount_point}; waiting.")
            else:
                try:
                    recording_dir.mkdir(parents=True, exist_ok=True)
                    if remaining() <= 0:
                        break
                    attempt += 1
                    filename = f"%Y%m%dT%H%M%SZ_osiris_{run_id}_{attempt:03d}.mp4"
                    command = [
                        str(args.ffmpeg), "-hide_banner", "-nostdin", "-nostats",
                        "-loglevel", "info", "-n", "-rtsp_transport", "tcp",
                        # FFmpeg 4.4 RTSP socket timeout, in microseconds.
                        "-stimeout", "15000000", "-fflags", "+genpts",
                        "-i", args.camera_url, "-map", "0:v:0", "-an", "-c:v", "copy",
                        "-f", "segment", "-segment_time", str(args.segment_seconds),
                        "-reset_timestamps", "1", "-strftime", "1",
                        str(recording_dir / filename),
                    ]
                    log(f"Opening camera stream, attempt {attempt}.")
                    process = subprocess.Popen(
                        command, env={**os.environ, "TZ": "UTC"}, start_new_session=True
                    )
                    try:
                        while remaining() > 0:
                            try:
                                status = process.wait(timeout=min(2, remaining()))
                                log(f"FFmpeg exited with status {status}.")
                                break
                            except subprocess.TimeoutExpired:
                                if not os.path.ismount(mount_point):
                                    log("NAS mount disappeared; stopping this attempt.")
                                    break
                    finally:
                        stop_ffmpeg(process)
                except OSError as error:
                    log(f"Recording attempt failed: {error}")

            if remaining() > 0:
                log("Retrying in up to 5 seconds.")
                time.sleep(min(5, remaining()))
    except KeyboardInterrupt:
        log("Recording stopped by user or termination signal.")
    finally:
        log(f"Recording session ended. Output directory: {recording_dir}")


if __name__ == "__main__":
    main()

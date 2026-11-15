#!/usr/bin/env python3

import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

FFMPEG = Path("/usr/local/Caskroom/miniforge/base/envs/walle/bin/ffmpeg")
CAMERA_URL = "rtsp://10.1.4.207:554/h264"

MOUNT_POINT = Path("/Volumes/oakley_lab")
NAS_URL = "smb://oakley@oakley-nas1/oakley_lab"
RECORDING_ROOT = MOUNT_POINT / "aquarium" / "2026Pan"

RECORDING_SECONDS = 12 * 60 * 60
SEGMENT_SECONDS = 30 * 60
RETRY_SECONDS = 5


def log(message):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{timestamp} {message}", flush=True)


def wait_for_nas(deadline):
    if os.path.ismount(MOUNT_POINT):
        return True

    log("NAS not mounted. Attempting to connect.")
    try:
        subprocess.run(["open", NAS_URL], check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        log(f"Could not open the NAS connection: {error}")

    mount_deadline = min(deadline, time.monotonic() + 60)
    while time.monotonic() < mount_deadline:
        if os.path.ismount(MOUNT_POINT):
            log("NAS mounted.")
            return True
        time.sleep(1)

    return False


def stop_ffmpeg(process):
    if process.poll() is not None:
        return

    # Give FFmpeg time to finalize the current MP4.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log("FFmpeg did not shut down within 10 seconds. Stopping it.")
        process.kill()
        process.wait()


def main():
    if not os.access(FFMPEG, os.X_OK):
        raise SystemExit(f"FFmpeg is missing or not executable: {FFMPEG}")

    # Use the local date at startup for the folder and UTC for clip filenames.
    recording_dir = RECORDING_ROOT / datetime.now().strftime("%Y%m%d")
    ffmpeg_env = {**os.environ, "TZ": "UTC"}
    run_id = uuid4().hex[:8]
    deadline = time.monotonic() + RECORDING_SECONDS
    attempt = 0

    log(f"Recording directory: {recording_dir}")
    log("Starting 12-hour recording. Press Ctrl-C to stop.")

    try:
        while time.monotonic() < deadline:
            if wait_for_nas(deadline):
                recording_dir.mkdir(parents=True, exist_ok=True)
                if time.monotonic() >= deadline:
                    break

                attempt += 1
                filename = f"%Y%m%dT%H%M%SZ_{run_id}_{attempt:03d}.mp4"
                command = [
                    str(FFMPEG),
                    "-hide_banner",
                    "-nostdin",
                    "-nostats",
                    "-loglevel", "info",
                    "-n",
                    "-rtsp_transport", "tcp",
                    "-i", CAMERA_URL,
                    "-map", "0:v:0",
                    "-c:v", "copy",
                    "-f", "segment",
                    "-segment_time", str(SEGMENT_SECONDS),
                    "-reset_timestamps", "1",
                    "-strftime", "1",
                    str(recording_dir / filename),
                ]

                log(f"Opening camera stream, attempt {attempt}.")
                process = subprocess.Popen(
                    command,
                    env=ffmpeg_env,
                    start_new_session=True,
                )
                try:
                    status = process.wait(
                        timeout=max(0, deadline - time.monotonic())
                    )
                    log(f"FFmpeg exited with status {status}.")
                except subprocess.TimeoutExpired:
                    log("Reached the 12-hour recording deadline.")
                    break
                finally:
                    stop_ffmpeg(process)
            else:
                log("NAS is unavailable.")

            remaining = deadline - time.monotonic()
            if remaining > 0:
                log("Retrying in up to 5 seconds.")
                time.sleep(min(RETRY_SECONDS, remaining))

    except KeyboardInterrupt:
        log("Recording stopped by user.")
    finally:
        log(f"Recording session ended. Output directory: {recording_dir}")


if __name__ == "__main__":
    main()

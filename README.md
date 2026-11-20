# MANTA

**Marine Animal Nocturnal Tracking and Analysis**

MANTA is an open toolkit for nocturnal marine-animal tracking and quantitative behavioral analysis, developed primarily for bioluminescent marine invertebrates using stereo, infrared, and low-light video.

The first teaching workflow demonstrates practical OpenCV calibration for a synchronized stereo-camera pair inside underwater globe housings.

The calibration model is an empirical approximation: it absorbs lens and housing effects into OpenCV's pinhole-plus-distortion model. Calibrate **underwater, with the cameras fixed in their final housings**, and validate across the image and working-distance range before using reconstructed distances quantitatively.

## Workflow

1. Supply synchronized left and right checkerboard videos.
2. Detect frames where both cameras see the checkerboard.
3. Measure board size, position, border clearance, and motion.
4. Filter and select a spatially diverse set of stable, nearby boards.
5. Estimate each camera's intrinsics and the stereo geometry.
6. Save a portable `.npz` calibration file.
7. Rectify held-out frames and inspect vertical disparity and epipolar alignment.
8. Rectify complete videos.

Two parallel teaching notebooks use real underwater calibration recordings:

- `01_stereo_rectification_walle.ipynb` calibrates separate synchronized left
  and right videos from South Water Caye 2019.
- `02_stereo_rectification_smalle.ipynb` splits a single side-by-side SMALL-E
  recording from Bocas 2024 in memory before applying the same workflow.

The large source videos and generated calibration artifacts are not copied into
this Git repository.

Start with the [WALL-E notebook](notebooks/01_stereo_rectification_walle.ipynb)
or the [SMALL-E notebook](notebooks/02_stereo_rectification_smalle.ipynb).

## Planned scope

- Stereo calibration, rectification, and 3-D reconstruction
- Infrared and low-light animal tracking
- Bioluminescent pulse detection and signal segmentation
- Behavioral summaries, statistical analysis, and visualization
- Reproducible student-facing notebooks and example datasets

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Input requirements

- Left and right videos must have equal frame size and frame rate.
- Videos must already be synchronized. `frame_offset` can compensate for a fixed whole-frame offset.
- `checkerboard_size` means the number of **internal corners**, not squares.
- `square_size` can use millimetres, centimetres, or another unit; reconstructed translations use that same unit.
- Record the board underwater through the final globe housings.
- Include varied board positions, orientations, and distances while keeping the complete board visible.

## Origin

This repository began as a cleaned teaching adaptation of calibration experiments from the WALL-E project. It now also includes the physiology pulse-discovery workflow described below.

## Physiology pulse discovery — Bocas 2026

[The physiology discovery notebook](notebooks/physiology_pulse_discovery_Bocas2026.ipynb)
searches editable vessel regions at native resolution for nearly stationary light
events. It saves frame/blob CSVs, candidate durations, fixed-site intensity traces,
and a short list of cropped review videos. Start with the local pilot, inspect the
provisional vessel-group outlines, then use the small time-spread Box batch.
Candidates and recurrent fiber associations require visual review; they are not
confirmed bioluminescence. Results are stored under
`data/processed/physiology_pulse_discovery_Bocas2026/`.

For the detailed full-night pass on the 2026-08-31 and 2026-09-02 observation
nights, use the resumable runner. It inventories the shared Box folder,
calibrates the fixed vessel regions from representative recordings, then
downloads and scans one recording at a time:

```bash
python scripts/physiologyFullNights.py inventory
python scripts/physiologyFullNights.py prepare
python scripts/physiologyFullNights.py run --workers 8
```

The live report and per-video blob tables are written to
`data/processed/physiology_full_nights_Bocas2026/`. The strict train search
requires at least three separate, sustained pulses with a net downward path;
exploratory train tables are retained separately for tuning.

Run the detector checks in the `manta` environment with
`python -m unittest discover -s tests -v`.

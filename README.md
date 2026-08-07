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

This repository is a cleaned teaching adaptation of calibration experiments from the WALL-E project. It intentionally excludes the later bioluminescent-pulse detection workflow.

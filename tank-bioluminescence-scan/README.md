# Bioluminescence scan data

The scanner and nightly notebook keep full-size intermediate data on the Oakley
NAS, not in Git. Mount the share at `~/labdata` on Linux or
`/Volumes/oakley_lab` on macOS before running the notebook or scanner without an
explicit `--output` path. The default NAS output directory is
`analysis/manta-analysis/tank-bioluminescence-scan/output/` relative to the
mounted share. If the share is not mounted, the scanner exits with an error
instead of writing large files into the repository.

The NAS output directory contains raw and fully classified detections,
combined CSVs, lighting records, and sand-reference masks. The repository
keeps only the curated prospective detections in
`results/bioluminescence_candidates.csv` (`bottom_bioluminescence` and
`uncertain` classes), along with compact provenance in `provenance/`. Sand
region summaries and the simulator log are retained; per-night mask images and
full reflection classifications stay on the NAS.

The notebook refreshes the candidate CSV and provenance files when it
classifies or reclassifies nights. Keep the NAS mounted while running it.

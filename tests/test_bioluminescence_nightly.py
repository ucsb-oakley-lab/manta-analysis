import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tank-bioluminescence-scan"
    / "nightly_bioluminescence.py"
)
spec = importlib.util.spec_from_file_location("nightly_bioluminescence", MODULE_PATH)
nightly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nightly)


class NightlyBioluminescenceTests(unittest.TestCase):
    def test_default_nas_output_requires_a_mounted_share(self):
        with patch.object(Path, "is_mount", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Oakley NAS is not mounted"):
                nightly.default_nas_output_dir()

    def test_split_and_rebuild_preserve_rows_without_reingesting_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            raw_master = output_dir / "bioluminescence_hits.csv"
            classified_master = output_dir / "bioluminescence_hits_classified.csv"
            raw_rows = [
                ["20261112", "video-b", "2"],
                ["20261111", "video-a", "1"],
            ]
            classified_rows = [row + ["uncertain"] for row in raw_rows]
            header = ["night", "video", "frame"]

            for path, rows, columns in (
                (raw_master, raw_rows, header),
                (classified_master, classified_rows, header + ["classification"]),
            ):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(rows)

            self.assertEqual(
                len(nightly.split_master_csv_by_night(raw_master, output_dir)), 2
            )
            self.assertEqual(
                len(
                    nightly.split_master_csv_by_night(
                        classified_master, output_dir, classified=True
                    )
                ),
                2,
            )
            self.assertEqual(
                nightly.rebuild_master_csv(output_dir, raw_master), len(raw_rows)
            )
            self.assertEqual(
                nightly.rebuild_master_csv(
                    output_dir, classified_master, classified=True
                ),
                len(classified_rows),
            )

            with raw_master.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream))[1:], raw_rows[::-1])
            with classified_master.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream))[1:], classified_rows[::-1])

    def test_scanning_a_night_invalidates_only_its_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            staged_csv = output_dir / "staged.csv"
            classified_csv = nightly.night_csv_path(
                output_dir, "20261118", classified=True
            )
            classified_csv.write_text("stale", encoding="utf-8")
            with staged_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video", "frame"])
                writer.writerow(["video-a", "4"])

            raw_csv, row_count = nightly.write_scanned_night(
                staged_csv, output_dir, "20261118"
            )

            self.assertEqual(row_count, 1)
            self.assertFalse(classified_csv.exists())
            with raw_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(
                rows,
                [["video", "frame", "night"], ["video-a", "4", "20261118"]],
            )

    def test_split_adds_night_from_legacy_video_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            master_csv = output_dir / "bioluminescence_hits.csv"
            with master_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video", "frame"])
                writer.writerow([r"Z:\aquarium\20261118\video.mp4", "12"])

            paths = nightly.split_master_csv_by_night(master_csv, output_dir)

            self.assertEqual([path.name for path in paths], ["20261118_hits.csv"])
            with paths[0].open(newline="", encoding="utf-8") as stream:
                self.assertEqual(
                    list(csv.reader(stream)),
                    [
                        ["video", "frame", "night"],
                        [r"Z:\aquarium\20261118\video.mp4", "12", "20261118"],
                    ],
                )

    def test_worklist_includes_raw_nights_without_classified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            raw_paths = [
                nightly.night_csv_path(output_dir, night)
                for night in ("20261118", "20261119")
            ]
            for path in raw_paths:
                path.touch()
            nightly.night_csv_path(
                output_dir, "20261118", classified=True
            ).touch()
            (output_dir / "bioluminescence_hits.csv").touch()

            self.assertEqual(
                nightly.classification_worklist(output_dir),
                [
                    (
                        "20261119",
                        raw_paths[1],
                        nightly.night_csv_path(
                            output_dir, "20261119", classified=True
                        ),
                    )
                ],
            )

    def test_candidate_csv_keeps_only_prospective_classifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "nas-output"
            candidate_csv = Path(tmp) / "repo" / "bioluminescence_candidates.csv"
            header = ["night", "video", "classification"]
            rows_by_night = {
                "20261118": [
                    ["20261118", "video-a", "bottom_bioluminescence"],
                    ["20261118", "video-a", "reflection_suspect"],
                ],
                "20261119": [
                    ["20261119", "video-b", "uncertain"],
                    ["20261119", "video-b", "excluded_artifact"],
                ],
            }
            for night, rows in rows_by_night.items():
                classified_csv = nightly.night_csv_path(
                    output_dir, night, classified=True
                )
                classified_csv.parent.mkdir(parents=True, exist_ok=True)
                with classified_csv.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(rows)

            candidate_csv.parent.mkdir(parents=True, exist_ok=True)
            candidate_csv.write_text("stale data", encoding="utf-8")
            row_count = nightly.rebuild_candidate_csv(output_dir, candidate_csv)

            with candidate_csv.open(newline="", encoding="utf-8") as stream:
                result = list(csv.reader(stream))
            self.assertEqual(row_count, 2)
            self.assertEqual(
                result,
                [
                    header,
                    ["20261118", "video-a", "bottom_bioluminescence"],
                    ["20261119", "video-b", "uncertain"],
                ],
            )

    def test_candidate_csv_requires_classified_night_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                nightly.rebuild_candidate_csv(Path(tmp), Path(tmp) / "candidates.csv")


if __name__ == "__main__":
    unittest.main()

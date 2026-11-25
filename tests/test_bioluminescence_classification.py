import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tank-bioluminescence-scan"
    / "classify_bioluminescence.py"
)
sys.path.insert(0, str(CLASSIFIER_PATH.parent))
spec = importlib.util.spec_from_file_location("classify_bioluminescence", CLASSIFIER_PATH)
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


class ArtifactClassificationTests(unittest.TestCase):
    def test_empty_sand_regions_csv_keeps_a_readable_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            regions_csv = Path(tmp) / "sand_regions.csv"

            classifier.write_sand_regions([], regions_csv)

            regions = pd.read_csv(regions_csv)

        self.assertTrue(regions.empty)
        self.assertEqual(list(regions.columns), classifier.SAND_REGION_FIELDS)

    def test_moon_state_uses_simulator_timezone_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            simulator_log = Path(tmp) / "simulator_log.csv"
            pd.DataFrame(
                [
                    {
                        "actual_check_time": "2026-11-18T15:28:00-08:00",
                        "simulator_time_at_check": "2026-11-18 15:28:00",
                        "simulator_latitude": 0.0,
                        "simulator_longitude": -82.25,
                        "simulator_timezone": "America/Panama",
                        "simulator_runs_continuously": True,
                    }
                ]
            ).to_csv(simulator_log, index=False)
            detections = pd.DataFrame(
                [
                    {
                        "video": "20261119T015509Z_b779144d_001.mp4",
                        "time_sec": 312.388,
                    }
                ]
            )

            result = classifier.add_moon_state_metadata(detections, simulator_log)

            log_without_timezone = Path(tmp) / "simulator_log_without_timezone.csv"
            pd.DataFrame(
                [
                    {
                        "actual_check_time": "2026-11-18T15:28:00-08:00",
                        "simulator_time_at_check": "2026-11-18 15:28:00",
                        "simulator_latitude": 0.0,
                        "simulator_longitude": -82.25,
                    }
                ]
            ).to_csv(log_without_timezone, index=False)
            default_result = classifier.add_moon_state_metadata(
                detections, log_without_timezone
            )

        self.assertEqual(
            result.loc[0, "actual_detection_time_utc"],
            "2026-11-19T02:00:21.388000+00:00",
        )
        self.assertEqual(
            result.loc[0, "inferred_simulator_time"],
            "2026-11-18 18:00:21.388000",
        )
        self.assertEqual(result.loc[0, "simulator_timezone"], "America/Panama")
        self.assertEqual(
            default_result.loc[0, "simulator_timezone"], "America/Panama"
        )

    def test_out_of_tank_roi_is_labeled_without_dropping_rows(self):
        detections = pd.DataFrame(
            [
                {
                    "night": "20261118",
                    "video": "video-a",
                    "frame": 1,
                    "time_sec": 1.0,
                    "x": 120,
                    "y": 600,
                    "area": 10,
                    "mean_intensity": 150,
                    "max_intensity": 200,
                    "bbox_width": 5,
                    "bbox_height": 5,
                },
                {
                    "night": "20261118",
                    "video": "video-a",
                    "frame": 2,
                    "time_sec": 2.0,
                    "x": 200,
                    "y": 600,
                    "area": 10,
                    "mean_intensity": 150,
                    "max_intensity": 200,
                    "bbox_width": 5,
                    "bbox_height": 5,
                },
            ]
        )

        result = classifier.classify(detections, {}, {}, min_track_span=30.0)

        self.assertEqual(len(result), len(detections))
        self.assertEqual(result.loc[0, "classification"], "excluded_artifact")
        self.assertEqual(
            result.loc[0, "classification_reason"],
            "fixed out-of-tank artifact ROI",
        )
        self.assertNotEqual(result.loc[1, "classification"], "excluded_artifact")


if __name__ == "__main__":
    unittest.main()

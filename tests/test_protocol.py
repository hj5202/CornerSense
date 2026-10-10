"""Test direction thresholds, range splitting, labels and evaluation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from cornersense.evaluation import evaluate_curve, protocol_label, summarize_results


class ProtocolTests(unittest.TestCase):
    def test_approaching_halves(self):
        t = np.linspace(0, 10, 101)
        result = evaluate_curve(protocol_label("csi_0319_class_approaching_all_m1"), 0.5 + .1*t, t, 0, 10)
        self.assertEqual(result["predicted_motion"], "approaching")
        self.assertEqual(result["predicted_regions"], ["R2", "R1"])
        self.assertTrue(result["both_correct"])

    def test_away_halves(self):
        t = np.linspace(0, 10, 101)
        result = evaluate_curve(protocol_label("csi_0319_class_movingaway_all_m1"), 1.5 - .1*t, t, 0, 10)
        self.assertEqual(result["predicted_motion"], "moving_away")
        self.assertEqual(result["predicted_regions"], ["R1", "R2"])
        self.assertTrue(result["both_correct"])

    def test_raw_midpoint_coverage(self):
        t = np.linspace(6, 10, 100)
        result = evaluate_curve(protocol_label("csi_0319_class_approaching_all_m1"), t*.1, t, 0, 10)
        self.assertIsNone(result["region_correct"])
        self.assertEqual(result["range_status"], "missing-physical-half-coverage")

    def test_fixed_range_and_noperson(self):
        t = np.arange(20.)
        result = evaluate_curve(protocol_label("1013_stage_approaching_r2_m1"), np.full(20, .85), t, 0, 19)
        self.assertEqual(result["predicted_motion"], "noperson")
        self.assertEqual(result["predicted_regions"], ["R2"])

    def test_noperson_includes_exact_direction_thresholds(self):
        t = np.arange(20.)
        outside = np.nextafter(.10, np.inf)
        for first, last, expected in ((0., .10, "noperson"),
                                      (.10, 0., "noperson"),
                                      (0., outside, "approaching"),
                                      (outside, 0., "moving_away")):
            with self.subTest(first=first, last=last):
                distance = np.repeat([first, last], 10)
                result = evaluate_curve({"motion": "unknown"}, distance, t, 0, 19)
                self.assertEqual(result["Ig"], last - first)
                self.assertEqual(result["predicted_motion"], expected)

    def test_noperson_prediction_is_independent_of_label(self):
        t = np.arange(20.)
        for motion, correct in (("approaching", False), ("moving_away", False),
                                ("no_person", True), ("unknown", None)):
            with self.subTest(motion=motion):
                result = evaluate_curve({"motion": motion}, np.ones(20), t, 0, 19)
                self.assertEqual(result["predicted_motion"], "noperson")
                self.assertIs(result["motion_correct"], correct)

    def test_unlabeled_curves_keep_predictions_without_scores(self):
        t = np.arange(20.)
        for label in (protocol_label("my_recording"), {}, {"motion": None}):
            for distance, prediction in ((np.linspace(.5, 1.5, 20), "approaching"),
                                         (np.linspace(1.5, .5, 20), "moving_away"),
                                         (np.ones(20), "noperson")):
                with self.subTest(label=label, prediction=prediction):
                    result = evaluate_curve(label, distance, t, 0, 19)
                    self.assertEqual(result["predicted_motion"], prediction)
                    for field in ("motion_correct", "region_correct", "both_correct"):
                        self.assertIsNone(result[field])
                    summary = summarize_results([{"status": "ok", "evaluation": result}])
                    self.assertEqual(summary["processing_ok"], 1)
                    for field in ("motion_total", "range_total", "both_total"):
                        self.assertEqual(summary[field], 0)

    def test_range_scoring_with_missing_motion_label(self):
        t = np.arange(20.)
        for region, origin, correct in (("R1", "explicit", True),
                                        ("R2", "explicit", False),
                                        ("temporal_half", "temporal_half", None)):
            with self.subTest(region=region):
                label = {"motion": "unknown", "region": region, "region_origin": origin}
                result = evaluate_curve(label, np.ones(20), t, 0, 19)
                self.assertIsNone(result["motion_correct"])
                self.assertIs(result["region_correct"], correct)
                self.assertIsNone(result["both_correct"])

    def test_no_person_and_robot_motion_share_negative_control(self):
        t = np.arange(20.)
        for name in ("csi0103_class_noperson_all_m1", "csi_0319_class_noperson_all_m1",
                     "csi_0325_class_noperson_all_m1", "1013_corridor_noperson_all_m1"):
            with self.subTest(name=name):
                label = protocol_label(name)
                self.assertEqual(label["motion"], "no_person")
                self.assertIsNone(label["region"])
                self.assertIsNone(label["region_origin"])
                result = evaluate_curve(label, np.ones(20), t, 0, 19)
                self.assertEqual(result["predicted_motion"], "noperson")
                self.assertTrue(result["motion_correct"])
                self.assertIsNone(result["gt_region"])
                self.assertEqual(result["predicted_regions"], [])
                self.assertIsNone(result["region_correct"])
                self.assertIsNone(result["both_correct"])

    def test_no_person_direction_and_range_not_inferred_from_label(self):
        t = np.arange(20.)
        label = dict(motion="no_person", region="temporal_half",
                     region_origin="temporal_half")
        result = evaluate_curve(label, np.linspace(.5, 1.5, 20), t, 0, 19)
        self.assertEqual(result["predicted_motion"], "approaching")
        self.assertFalse(result["motion_correct"])
        self.assertIsNone(result["gt_region"])
        self.assertEqual(result["predicted_regions"], [])
        self.assertIsNone(result["region_correct"])

    def test_labels_do_not_change_direction(self):
        t = np.arange(20.)
        d = np.linspace(.5, 1.5, 20)
        a = evaluate_curve(protocol_label("csi_0319_class_approaching_all_m1"), d, t, 0, 19)
        b = evaluate_curve(protocol_label("csi_0319_class_movingaway_all_m1"), d, t, 0, 19)
        for key in ("Ig", "tr_min_m", "predicted_motion", "predicted_regions"):
            self.assertEqual(a[key], b[key])

    def test_summary_keeps_no_person_out_of_range_denominators(self):
        t = np.arange(20.)
        rows = [{"status": "ok", "evaluation": evaluate_curve(
                    protocol_label(name), distance, t, 0, 19)}
                for name, distance in (
                    ("csi_0319_class_approaching_all_m1", np.linspace(.5, 1.5, 20)),
                    ("csi_0319_class_noperson_all_m1", np.ones(20)),
                    ("my_recording", np.linspace(.5, 1.5, 20)))]
        rows.append({"status": "error", "error": "No curve"})
        self.assertEqual(summarize_results(rows), {
            "total": 4, "processing_ok": 3, "motion_total": 2, "motion_correct": 2,
            "range_total": 1, "region_correct": 1, "both_total": 1, "both_correct": 1,
            "noperson_total": 1, "noperson_correct": 1})

    def test_filename_must_match_complete_sample_pattern(self):
        for name in ("csi_0319_class_approaching_all_m1_extra",
                     "csi0103_stage_movingaway_r2_m1_extra",
                     "csi_0325_class_approaching_all_m1.txt",
                     "csi_0319_class_still_all_m1"):
            self.assertEqual(protocol_label(name)["motion"], "unknown")


if __name__ == "__main__":
    unittest.main()

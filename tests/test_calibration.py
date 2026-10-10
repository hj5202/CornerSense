"""Test fixed session offsets, validation and per-folder runtime routing."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cornersense.evaluation import evaluate_curve, protocol_label
from cornersense.inference import calibrate_angles, load_calibration, validate_calibration


class CalibrationTests(unittest.TestCase):
    def test_published_fixed_session_offsets(self):
        calibration = load_calibration(ROOT / "calibration/sample_angles.json")
        self.assertEqual(calibration, {"method": "fixed_session_offset", "offsets_deg": {
            "0319_data": -3., "0325_data": 22., "data_1013": -12., "data_sup01": 0.}})

    def test_same_scalar_is_subtracted_from_both_paths(self):
        measured = {"theta_r": 30., "theta_u": -20.}
        for offset in (-3., 22., -12., 0.):
            with self.subTest(offset=offset):
                result = calibrate_angles(measured, offset)
                self.assertEqual(result, {key: value - offset for key, value in measured.items()})
                self.assertEqual(result["theta_r"] - result["theta_u"], 50.)
        self.assertEqual(measured, {"theta_r": 30., "theta_u": -20.})

    def test_out_of_range_angles_are_rejected_without_clipping(self):
        self.assertEqual(calibrate_angles({"theta_r": 80., "theta_u": -80.}, 0.),
                         {"theta_r": 80., "theta_u": -80.})
        for measured, offset in (({"theta_r": 80., "theta_u": 0.}, -1.),
                                 ({"theta_r": 0., "theta_u": -80.}, 1.)):
            with self.subTest(measured=measured, offset=offset):
                with self.assertRaisesRegex(ValueError, "Corrected AoA"):
                    calibrate_angles(measured, offset)

    def test_nonfinite_angles_and_offsets_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    calibrate_angles({"theta_r": value, "theta_u": 0.}, 0.)
                with self.assertRaises(ValueError):
                    calibrate_angles({"theta_r": 0., "theta_u": value}, 0.)
        for offset in (True, "3", np.nan, np.inf, -np.inf, 1j):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    calibrate_angles({"theta_r": 0., "theta_u": 0.}, offset)

    def test_invalid_configuration_is_rejected(self):
        invalid = [None, {}, {"method": "old", "offsets_deg": {"a": 0}},
                   {"method": "fixed_session_offset", "offsets_deg": {}},
                   {"method": "fixed_session_offset", "offsets_deg": []}]
        invalid.extend({"method": "fixed_session_offset", "offsets_deg": {key: 0}}
                       for key in ("", ".", "..", "a/b", "a\\b", 3))
        invalid.extend({"method": "fixed_session_offset", "offsets_deg": {"a": value}}
                       for value in (True, "3", None, np.nan, np.inf))
        for document in invalid:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    validate_calibration(document)

    @staticmethod
    def fake_inference(path, offset):
        path = Path(path)
        times = np.arange(20.)
        distance = np.ones(20)
        measured = {"theta_r": 30., "theta_u": 20.}
        row = {"group": path.parent.name, "filename": path.name, "status": "ok",
               "calibration_offset_deg": float(offset), "measured_aoa": measured,
               "selected_aoa": calibrate_angles(measured, offset),
               "trace_extent_s": [0., 19.],
               "evaluation": evaluate_curve(protocol_label(path.name), distance, times, 0, 19)}
        return row, times, distance

    def test_same_folder_uses_one_offset(self):
        import run
        offsets = {"0319_data": -3., "0325_data": 22.}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, "infer", side_effect=self.fake_inference) as infer, \
                patch.object(run, "plot_file"):
            root = Path(directory)
            names = ("csi_0319_class_noperson_all_m1", "csi_0319_class_approaching_all_m1")
            rows = [run.process_file((root / "0319_data" / name, root / "outputs", offsets))
                    for name in names]
            self.assertEqual([call.args[1] for call in infer.call_args_list], [-3., -3.])
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertEqual(rows[0]["selected_aoa"], rows[1]["selected_aoa"])

    def test_missing_session_is_rejected_before_inference(self):
        import run
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(run, "infer") as infer,
              redirect_stderr(io.StringIO())):
            root = Path(directory)
            row = run.process_file((root / "unknown" / "recording", root / "outputs",
                                    {"0319_data": -3.}))
            self.assertEqual(row["status"], "error")
            self.assertIn("unknown", row["error"])
            infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Test result checks in normal and optimized Python execution.

Checks cover scalar/list comparisons and rejection of invalid result metadata.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify
from verify import compare
from cornersense.evaluation import evaluate_curve, protocol_label, summarize_results
from cornersense.inference import calibrate_angles
from cornersense.metric_units import metric_units, phase_evaluation, to_paper_g


class VerificationTests(unittest.TestCase):
    @staticmethod
    def make_saved_result(root):
        group = "0319_data"
        filename = "csi_0319_class_noperson_all_m1"
        recording = root / "sample" / group / filename
        recording.parent.mkdir(parents=True)
        recording.write_bytes(b"raw recording")
        output = root / "results"
        folder = output / group / filename
        folder.mkdir(parents=True)
        time = np.arange(20.)
        distance = np.ones(20)
        np.savez_compressed(folder / "curve.npz", time_s=time, Dphys_m=distance,
                            tr_m=distance, G_rad=to_paper_g(distance))
        for name in ("metrics.png", "metrics.svg"):
            (folder / name).write_bytes(b"x" * 1001)
        measured = {"theta_r": 30., "theta_u": 20.}
        row = {"group": group, "filename": filename, "status": "ok",
               "calibration_offset_deg": -3., "measured_aoa": measured,
               "selected_aoa": calibrate_angles(measured, -3.), "trace_extent_s": [0., 19.],
               "evaluation": phase_evaluation(evaluate_curve(
                   protocol_label(filename), distance, time, 0, 19)),
               "artifacts": {key: f"{group}/{filename}/{value}" for key, value in
                             {"curve": "curve.npz", "plot": "metrics.png",
                              "vector_plot": "metrics.svg"}.items()}}
        (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")
        document = {"metadata": {"metric_units": metric_units(),
                    "calibration": {"method": "fixed_session_offset",
                                    "offsets_deg": {group: -3.}}},
                    "results": [row], "summary": summarize_results([row])}
        path = output / "results.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path, document

    def test_result_verification_uses_saved_calibration_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.make_saved_result(root)
            calibration = root / "calibration" / "sample_angles.json"
            calibration.parent.mkdir()
            calibration.write_text(json.dumps({"method": "fixed_session_offset",
                "offsets_deg": {"0319_data": 7.}}), encoding="utf-8")
            with patch.object(verify, "ROOT", root):
                report = verify.verify_results(path)
            self.assertEqual(report["motion_correct"], 1)
            self.assertEqual(report["range_total"], 0)
            self.assertEqual(report["noperson_correct"], 1)

    def test_saved_offset_must_match_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document = self.make_saved_result(root)
            document["results"][0]["calibration_offset_deg"] = 0.
            path.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(verify, "ROOT", root), \
                    self.assertRaisesRegex(ValueError, "calibration offset"):
                verify.verify_results(path)

    def test_result_summary_checks_applicable_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document = self.make_saved_result(root)
            document["summary"]["range_total"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(verify, "ROOT", root), \
                    self.assertRaisesRegex(ValueError, "Recomputed counts"):
                verify.verify_results(path)

    def test_comparisons_keep_the_existing_tolerance(self):
        compare(1. + 1e-10, 1., 'number')
        compare([True, 'R1', .85], [True, 'R1', .85], 'values')
        for actual, expected in ((1., 2.), ([1], [1, 2]), (False, True), ('R1', 'R2')):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(ValueError):
                    compare(actual, expected, 'probe')

    def test_optimized_python_rejects_a_numeric_mismatch(self):
        run = subprocess.run(
            [sys.executable, '-B', '-O', '-c',
             "from verify import compare; compare(1, 2, 'probe')"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('ValueError: probe', run.stderr)

    def test_optimized_cli_rejects_invalid_metric_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps({'metadata': {'metric_units': {}}}), encoding='utf-8')
            run = subprocess.run(
                [sys.executable, '-B', '-O', str(ROOT / 'verify.py'), '--results', str(path)],
                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Result metric units', run.stderr)
        self.assertNotIn('"status": "passed"', run.stdout)


if __name__ == '__main__':
    unittest.main()

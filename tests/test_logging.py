"""Check that inference logs use the returned motion and range evaluation."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from cornersense import inference


class EvaluationLoggingTests(unittest.TestCase):
    def run_inference(self, filename, distance, times=None):
        times = np.arange(len(distance), dtype=float) if times is None else times
        csi = np.ones((len(times), 3, 30), dtype=complex)
        core = {"Dphys": np.asarray(distance), "metric_t": times}
        log = io.StringIO()
        with patch.object(inference, "decode", return_value=(csi, times)), \
                patch.object(inference, "measure_angles",
                             return_value={"theta_r": 35., "theta_u": 25.}), \
                patch.object(inference.pipeline, "run_file", return_value=core), \
                contextlib.redirect_stdout(log):
            row, _, _ = inference.infer(filename, 0.)
        return row["evaluation"], log.getvalue()

    def test_temporal_range_log_matches_evaluation(self):
        evaluation, log = self.run_inference(
            "csi_0319_class_movingaway_all_m1", np.linspace(1.5, .5, 20))
        self.assertEqual(evaluation["predicted_regions"], ["R1", "R2"])
        self.assertIn("motion=moving_away; range=R1 -> R2", log)

    def test_no_person_log_marks_range_not_applicable(self):
        evaluation, log = self.run_inference(
            "csi_0319_class_noperson_all_m1", np.ones(20))
        self.assertEqual(evaluation["predicted_motion"], "noperson")
        self.assertEqual(evaluation["predicted_regions"], [])
        self.assertIn("motion=noperson; range=not applicable", log)

    def test_unknown_label_log_marks_range_unavailable(self):
        evaluation, log = self.run_inference(
            "new_session_recording", np.ones(20))
        self.assertEqual(evaluation["predicted_motion"], "noperson")
        self.assertEqual(evaluation["predicted_regions"], [])
        self.assertIn("motion=noperson; range=unavailable (unrecognized recording label)", log)


if __name__ == "__main__":
    unittest.main()

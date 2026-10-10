"""Check that batch processing exposes failures and warnings in the terminal."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run
from cornersense.evaluation import evaluate_curve, protocol_label


class RuntimeOutputTests(unittest.TestCase):
    @staticmethod
    def successful_inference(path, offset):
        print("Selected CSI segment")
        print("WARNING: diagnostic fallback")
        print("Numerical warning on stderr", file=sys.stderr)
        time = np.arange(20.)
        distance = np.ones(20)
        return {"group": path.parent.name, "filename": path.name, "status": "ok",
                "trace_extent_s": [0., 19.],
                "evaluation": evaluate_curve(protocol_label(path.name), distance, time, 0., 19.)}, time, distance

    def process(self, inference, verbose=False):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(run, "infer", side_effect=inference),
            patch.object(run, "plot_file"),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            root = Path(directory)
            row = run.process_file((root / "0319_data" / "csi_0319_class_noperson_all_m1",
                                    root / "output", {"0319_data": -3.}), verbose=verbose)
        return row, stdout.getvalue(), stderr.getvalue()

    def test_failure_prints_context_and_traceback(self):
        def failed_inference(path, offset):
            print("Decoding CSI packets")
            raise RuntimeError("bad packet")
        row, _, stderr = self.process(failed_inference)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error"], "bad packet")
        self.assertIn("Decoding CSI packets", stderr)
        self.assertIn("Traceback (most recent call last)", stderr)
        self.assertIn("RuntimeError: bad packet", stderr)

    def test_success_keeps_warnings_visible(self):
        row, stdout, stderr = self.process(self.successful_inference)
        self.assertEqual(row["status"], "ok")
        self.assertNotIn("Selected CSI segment", stdout)
        self.assertIn("WARNING: diagnostic fallback", stderr)
        self.assertIn("Numerical warning on stderr", stderr)

    def test_verbose_prints_processing_details(self):
        row, stdout, stderr = self.process(self.successful_inference, verbose=True)
        self.assertEqual(row["status"], "ok")
        self.assertIn("Selected CSI segment", stdout)
        self.assertIn("WARNING: diagnostic fallback", stdout)
        self.assertIn("Numerical warning on stderr", stderr)


if __name__ == "__main__":
    unittest.main()

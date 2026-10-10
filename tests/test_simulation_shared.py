"""Test the shared CSI simulation, pipeline integration and exported curves."""
import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cornersense import inference
from cornersense.core import constants, pipeline
from cornersense.simulation import run as runner
from cornersense.simulation.evaluate import compare_with_truth, display_scale, extract, metrics
from cornersense.simulation.generate import generate
from cornersense.simulation.pipeline import detect

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("simulation_2_dynamic_paths", "simulation_4_dynamic_paths")


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.inference, cls.constants = pipeline, inference, constants
        cls.examples = {}
        for name in PROJECTS:
            directory = ROOT / name
            config = json.loads((directory / "config.json").read_text())
            observation, truth = generate(config, cls.constants.FREQS, cls.constants.C)
            original = observation["csi"].copy()
            with patch.object(cls.inference, "measure_angles", wraps=cls.inference.measure_angles) as angles, \
                 patch.object(cls.pipeline, "run_file", wraps=cls.pipeline.run_file) as pipeline_run:
                detection = detect(observation["csi"], observation["time_s"], 0)
            with ExitStack() as stack:
                controls = {key: stack.enter_context(patch.object(cls.pipeline, key,
                    wraps=getattr(cls.pipeline, key))) for key in ("ma2d", "bandlimit_t", "stage_pca")}
                report, data, intermediate = extract(detection)
            cls.examples[name] = {"directory": directory, "config": config,
                "observation": observation, "truth": truth, "original": original,
                "detection": detection, "report": report, "data": data,
                "intermediate": intermediate, "angles_call": angles,
                "pipeline_call": pipeline_run, "control_calls": controls}

    def assert_tree_equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            np.testing.assert_array_equal(left, right)

    def test_sampling_and_frequency_grid(self):
        for name, item in self.examples.items():
            with self.subTest(project=name):
                observation = item["observation"]
                self.assertEqual(observation["csi"].shape, (3201, 3, 30))
                self.assertEqual(observation["time_s"][-1], 8)
                np.testing.assert_array_equal(observation["frequencies_hz"], self.constants.FREQS)

    def test_pipeline_warnings_and_failure_diagnostics_are_visible(self):
        def processing(*args, **kwargs):
            print("ordinary processing detail")
            print("WARNING: fallback used")
            return {}

        def failure(*args, **kwargs):
            print("failure context")
            raise ValueError("invalid synthetic input")

        errors = io.StringIO()
        with patch.object(self.inference, "measure_angles", return_value={"theta_r": 30, "theta_u": 60}), \
             patch.object(self.pipeline, "run_file", side_effect=processing), redirect_stderr(errors):
            detect(None, None, 0)
        self.assertIn("WARNING: fallback used", errors.getvalue())
        self.assertNotIn("ordinary processing detail", errors.getvalue())
        errors = io.StringIO()
        with patch.object(self.inference, "measure_angles", side_effect=failure), redirect_stderr(errors):
            with self.assertRaisesRegex(ValueError, "invalid synthetic input"):
                detect(None, None, 0)
        self.assertIn("failure context", errors.getvalue())

    def test_saved_results_cannot_be_overwritten(self):
        for item in self.examples.values():
            with self.assertRaises(FileExistsError):
                runner.run(item["directory"] / "config.json", item["directory"] / "results")

    def test_complex_path_superposition(self):
        for item in self.examples.values():
            np.testing.assert_array_equal(item["observation"]["csi"],
                item["truth"]["path_fields"].sum(axis=0) + item["truth"]["noise"])

    def test_deterministic_generator(self):
        for item in self.examples.values():
            again, _ = generate(item["config"], self.constants.FREQS, self.constants.C)
            np.testing.assert_array_equal(again["csi"], item["original"])

    def test_spatial_phase_convention(self):
        f = self.constants.FREQS[14]
        for item in self.examples.values():
            for index, path in enumerate(item["config"]["paths"]):
                fields = item["truth"]["path_fields"]
                ratio = fields[index, 0, 1, 14] / fields[index, 0, 0, 14]
                expected = np.exp(2j*np.pi*f*.025*np.sin(np.deg2rad(path["angle_deg"])) / self.constants.C)
                self.assertAlmostEqual(abs(ratio - expected), 0, places=10)

    def test_pipeline_calls_and_observation_input(self):
        for item in self.examples.values():
            item["angles_call"].assert_called_once()
            item["pipeline_call"].assert_called_once()
            arguments = item["pipeline_call"].call_args.kwargs
            self.assertTrue(arguments["return_intermediates"])
            self.assertIs(arguments["preloaded"][0], item["observation"]["csi"])
            self.assertIs(arguments["preloaded"][1], item["observation"]["time_s"])
            np.testing.assert_array_equal(item["observation"]["csi"], item["original"])

    def test_exact_returned_projection_arrays(self):
        for item in self.examples.values():
            mid = item["detection"]["result"]["intermediates"]
            for key in ("score", "loading", "singular_values"):
                np.testing.assert_array_equal(item["intermediate"][f"augmented_{key}"], mid["stage1"][key])
            np.testing.assert_array_equal(item["intermediate"]["augmented_processed"], mid["P2p"])

    def test_same_pipeline_preprocessing_for_both_powers(self):
        for item in self.examples.values():
            fs, intermediate = item["report"]["sample_rate_hz"], item["intermediate"]
            for method in ("augmented", "raw"):
                expected = self.pipeline.bandlimit_t(
                    self.pipeline.ma2d(intermediate[f"{method}_power_matrix"], fs, .2),
                    fs, self.pipeline.BAND_LO, self.pipeline.BAND_HI).real
                np.testing.assert_array_equal(expected, intermediate[f"{method}_processed"])
            for call in item["control_calls"].values():
                call.assert_called_once()

    def test_power_arrays_are_uncentered(self):
        for item in self.examples.values():
            intermediate, data = item["intermediate"], item["data"]
            h, ha = intermediate["Hs"][:, 0, 14], intermediate["Ha"]
            np.testing.assert_array_equal(data["raw_power"], np.abs(h)**2)
            np.testing.assert_array_equal(data["augmented_power"], np.abs(h + ha)**2)

    def test_projection_and_sign_invariance(self):
        for item in self.examples.values():
            for method in ("augmented", "raw"):
                z = item["intermediate"][f"{method}_score"]
                v = item["intermediate"][f"{method}_loading"][14]
                np.testing.assert_array_equal(item["intermediate"][f"{method}_projection"], z*v)
                np.testing.assert_array_equal(display_scale(z*v), display_scale((-z)*(-v)))

    def test_normalization_uses_own_curve(self):
        for item in self.examples.values():
            for method in ("augmented", "raw"):
                shown = item["data"][f"{method}_projection_display"]
                self.assertAlmostEqual(float(shown.mean()), 0, places=12)
                self.assertAlmostEqual(float(np.mean(shown**2)), .5, places=12)

    def test_reference_virtual_cosine_identity(self):
        for item in self.examples.values():
            report, data = compare_with_truth(copy.deepcopy(item["report"]),
                copy.deepcopy(item["data"]), item["intermediate"], item["truth"])
            start, end = report["segment_samples"]
            reference = list(item["truth"]["path_names"]).index("reference")
            phase = -np.angle(item["truth"]["path_fields"][reference, start:end, 0, 14])
            expected = np.cos(phase + np.angle(item["intermediate"]["Ha"]))
            np.testing.assert_allclose(data["truth_cos"], expected, rtol=0, atol=1e-15)

    def test_truth_is_used_after_projection_extraction(self):
        for item in self.examples.values():
            truth = copy.deepcopy(item["truth"])
            truth["path_fields"] *= np.exp(.4j)
            _, data = compare_with_truth(copy.deepcopy(item["report"]),
                copy.deepcopy(item["data"]), item["intermediate"], truth)
            for method in ("augmented", "raw"):
                key = f"{method}_projection_display"
                np.testing.assert_array_equal(item["data"][key], data[key])

    def test_nonfinite_comparison_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.array([np.nan]), np.array([0.]))

    def test_saved_curves_match_pipeline_run(self):
        for name, item in self.examples.items():
            with self.subTest(project=name):
                report, data = compare_with_truth(copy.deepcopy(item["report"]),
                    copy.deepcopy(item["data"]), item["intermediate"], item["truth"])
                with np.load(item["directory"] / "results/data.npz", allow_pickle=False) as saved:
                    for key, array in data.items():
                        np.testing.assert_allclose(array, saved[key], rtol=1e-7, atol=1e-9)
                saved_report = json.loads((item["directory"] / "results/report.json").read_text())
                for method in ("augmented", "raw"):
                    for metric in ("correlation", "rmse"):
                        self.assertAlmostEqual(report[f"{method}_vs_truth"][metric],
                            saved_report[f"{method}_vs_truth"][metric], delta=1e-7)

    def test_pipeline_and_projection_repeat_exactly(self):
        for name, item in self.examples.items():
            with self.subTest(project=name):
                observation = item["observation"]
                again = detect(observation["csi"], observation["time_s"], 0)
                report, data, intermediate = extract(again)
                self.assert_tree_equal(again, item["detection"])
                self.assert_tree_equal(report, item["report"])
                self.assert_tree_equal(data, item["data"])
                self.assert_tree_equal(intermediate, item["intermediate"])
                np.testing.assert_array_equal(observation["csi"], item["original"])

    def test_runner_processes_once_and_exports_scientific_report(self):
        with tempfile.TemporaryDirectory(prefix="cornersense-simulation-") as temporary, \
             patch.object(runner, "detect", wraps=runner.detect) as detector, \
             patch.object(runner, "extract", wraps=runner.extract) as extractor, \
             patch.object(runner, "make_figures") as plotter, redirect_stdout(io.StringIO()):
            output = Path(temporary) / "result"
            report = runner.run(ROOT / PROJECTS[0] / "config.json", output)
            detector.assert_called_once()
            extractor.assert_called_once()
            plotter.assert_called_once_with(output.resolve())
            self.assertEqual(report["processing_settings"], {"band_hz": [.8, 8.], "smoothing_s": .2})
            expected_files = {"config.json", "report.json", "observation.npz",
                              "truth.npz", "stage1.npz", "data.npz"}
            self.assertTrue(expected_files <= {p.name for p in output.iterdir()})
            self.assertEqual(json.loads((output / "report.json").read_text()),
                             runner.json_safe(report))


if __name__ == "__main__":
    unittest.main()

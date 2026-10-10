"""Test simulation entry points and package resolution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("simulation_2_dynamic_paths", "simulation_4_dynamic_paths")


class SimulationCliTests(unittest.TestCase):
    def invoke(self, arguments, *, cwd=ROOT, env=None):
        return subprocess.run([sys.executable, "-B", *arguments], cwd=cwd,
                              env=env, capture_output=True, text=True, timeout=60)

    def test_help_lists_configuration_and_output_options(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                result = self.invoke(["-m", f"{scenario}.run", "--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--config", result.stdout)
                self.assertIn("--output", result.stdout)

    def test_scripts_run_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix="cornersense-cli-") as temporary:
            directory = Path(temporary)
            environment = dict(os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                MPLCONFIGDIR=str(directory / "matplotlib"),
                XDG_CACHE_HOME=str(directory / "cache"))
            for scenario in SCENARIOS:
                with self.subTest(scenario=scenario):
                    output = directory / scenario
                    result = self.invoke([str(ROOT / scenario / "run.py"),
                        "--config", str(ROOT / scenario / "config.json"),
                        "--output", str(output)], cwd=directory, env=environment)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads((output / "report.json").read_text())
                    saved = json.loads((ROOT / scenario / "results/report.json").read_text())
                    for method in ("augmented", "raw"):
                        for metric in ("correlation", "rmse"):
                            self.assertAlmostEqual(report[f"{method}_vs_truth"][metric],
                                saved[f"{method}_vs_truth"][metric], delta=1e-7)
                    for figure in ("cosine_comparison", "power_comparison"):
                        self.assertTrue((output / "figures" / f"{figure}.png").is_file())
                        self.assertTrue((output / "figures" / f"{figure}.svg").is_file())


if __name__ == "__main__":
    unittest.main()

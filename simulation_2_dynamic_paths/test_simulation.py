"""Test the two-dynamic-path scenario settings."""
import json
from pathlib import Path
import unittest


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent / "config.json").read_text())

    def test_two_dynamic_paths_and_static_background(self):
        paths = self.config["paths"]
        self.assertEqual(len(paths), 3)
        dynamic = [p for p in paths if p["rate_mps"] != 0]
        static = [p for p in paths if p["rate_mps"] == 0]
        self.assertEqual([p["name"] for p in dynamic], ["reference", "secondary"])
        self.assertEqual(len(static), 1)
        self.assertEqual(static[0]["amplitude"], 0.1)

    def test_reference_parameters(self):
        reference = next(p for p in self.config["paths"] if p["name"] == "reference")
        self.assertEqual(reference, {"name": "reference", "length_m": 20,
            "rate_mps": -0.1, "amplitude": 1, "angle_deg": 30, "phase_rad": 0.5})


if __name__ == "__main__":
    unittest.main()

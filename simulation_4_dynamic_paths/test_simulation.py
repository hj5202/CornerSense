"""Test the four-dynamic-path scenario settings."""
import json
from pathlib import Path
import unittest

import numpy as np

from cornersense.core.constants import C, FREQS
from cornersense.simulation.generate import generate


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent / "config.json").read_text())

    def test_four_dynamic_paths_have_decreasing_amplitudes(self):
        paths = self.config["paths"]
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(p["rate_mps"] != 0 for p in paths))
        self.assertEqual(len({p["rate_mps"] for p in paths}), 4)
        self.assertTrue(np.all(np.diff([p["amplitude"] for p in paths]) < 0))

    def test_reference_path_is_preserved(self):
        self.assertEqual(self.config["paths"][0], {"name": "reference", "length_m": 20,
            "rate_mps": -0.1, "amplitude": 1, "angle_deg": 30, "phase_rad": 0.5})

    def test_paths_change_length_at_configured_rates(self):
        _, truth = generate(self.config, FREQS, C)
        for index, path in enumerate(self.config["paths"]):
            expected = path["length_m"] + path["rate_mps"] * truth["time_s"]
            np.testing.assert_array_equal(truth["path_lengths_m"][index], expected)
            self.assertNotEqual(expected[0], expected[-1])


if __name__ == "__main__":
    unittest.main()

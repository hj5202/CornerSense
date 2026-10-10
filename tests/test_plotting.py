"""Check that figures display formal range evaluations from in-memory curves."""
import copy
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from cornersense import plotting
from cornersense.evaluation import evaluate_curve
from cornersense.metric_units import to_paper_g


class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0, 10, 101)
        self.distance = np.linspace(.5, 1.5, 101)

    def evaluation(self, motion="approaching", region="R2", temporal=False):
        label = {"motion": motion, "region": region}
        if temporal:
            label["region_origin"] = "temporal_half"
        return evaluate_curve(label, self.distance, self.times, 0, 10)

    def capture(self, function, *args):
        with patch.object(plotting, "_save", side_effect=lambda fig, path: fig):
            figure = function("unused", *args)
        self.addCleanup(plt.close, figure)
        return figure

    def single(self, evaluation):
        return self.capture(plotting.plot_file, self.times, self.distance,
                            [0, 10], evaluation, "synthetic recording")

    def scene(self, *evaluations):
        rows = [dict(filename=f"recording {index}", status="ok",
                     evaluation=evaluation)
                for index, evaluation in enumerate(evaluations)]
        return self.capture(plotting.plot_scene, rows)

    def assert_unclassified(self, evaluation, reason):
        single = self.single(evaluation)
        self.assertIn(f"Range: {reason}", [item.get_text() for item in single.texts])
        self.assertFalse(any("Range decision:" in item.get_text()
                             for item in single.texts))
        self.assertFalse(any("T_R" in line.get_label()
                             for line in single.axes[0].lines))
        scene = self.scene(evaluation)
        range_axis = scene.axes[1]
        self.assertEqual(len(range_axis.patches), 0)
        self.assertEqual(len(range_axis.lines), 0)
        self.assertIn(reason.capitalize(), [item.get_text() for item in range_axis.texts])

    def test_no_person_range_is_not_applicable(self):
        for distance in (np.ones(101), self.distance):
            with self.subTest(moving=bool(np.ptp(distance))):
                evaluation = evaluate_curve({"motion": "no_person"}, distance,
                                            self.times, 0, 10)
                self.assert_unclassified(evaluation, "not applicable")

    def test_noperson_prediction_uses_exact_display_name(self):
        evaluation = evaluate_curve({"motion": "no_person"}, np.ones(101),
                                    self.times, 0, 10)
        self.assertEqual(evaluation["predicted_motion"], "noperson")
        single = self.single(evaluation)
        self.assertEqual(single.axes[0].get_title(), "noperson  |  Ig = +0.000 m")

    def test_no_person_ignores_stale_range_fields(self):
        evaluation = self.evaluation(motion="no_person")
        evaluation.update(predicted_regions=["R1"], tr_min_m=1.5)
        self.assert_unclassified(evaluation, "not applicable")

    def test_unknown_recording_label_has_distinct_explanation(self):
        evaluation = self.evaluation(motion="unknown", region=None)
        self.assert_unclassified(evaluation, "unavailable (unrecognized recording label)")

    def test_insufficient_half_coverage_has_distinct_explanation(self):
        evaluation = evaluate_curve(
            {"motion": "approaching", "region_origin": "temporal_half"},
            self.distance, np.linspace(6, 10, 101), 0, 10)
        self.assert_unclassified(evaluation,
                                 "unavailable (insufficient coverage of a trace half)")

    def test_fixed_range_uses_formal_label_and_converted_minimum(self):
        evaluation = self.evaluation()
        evaluation["predicted_regions"] = ["R1"]
        single = self.single(evaluation)
        self.assertIn("Range: R1 (min tr=0.500 m)",
                      [item.get_text() for item in single.texts])
        scene = self.scene(evaluation)
        axis = scene.axes[1]
        self.assertEqual(len(axis.patches), 1)
        self.assertAlmostEqual(axis.patches[0].get_width(), to_paper_g(.5))
        self.assertTrue(any(item.get_text().endswith("R1") for item in axis.texts))
        self.assertEqual(len(axis.lines), 1)

    def test_temporal_range_displays_both_formal_halves(self):
        evaluation = self.evaluation(region="temporal_half", temporal=True)
        original = copy.deepcopy(evaluation)
        single = self.single(evaluation)
        self.assertIn("Range: first half R2 (min tr=0.500 m); "
                      "second half R1 (min tr=1.000 m)",
                      [item.get_text() for item in single.texts])
        scene = self.scene(evaluation)
        axis = scene.axes[1]
        self.assertEqual(len(axis.patches), 2)
        np.testing.assert_allclose([bar.get_width() for bar in axis.patches],
                                   to_paper_g([.5, 1.]))
        self.assertTrue(any("H1: R2" in item.get_text() for item in axis.texts))
        self.assertTrue(any("H2: R1" in item.get_text() for item in axis.texts))
        self.assertEqual(evaluation, original)

    def test_missing_or_mismatched_range_results_are_not_inferred(self):
        fixed = self.evaluation()
        temporal = self.evaluation(region="temporal_half", temporal=True)
        cases = [
            dict(fixed, predicted_regions=[]),
            dict(fixed, tr_min_m=None),
            dict(fixed, predicted_regions=["R1", "R2"]),
            dict(fixed, predicted_regions=["unavailable"]),
            dict(temporal, predicted_regions=["R1"]),
            dict(temporal, half_tr_min_m=[]),
            dict(temporal, half_tr_min_m=[.5]),
            dict(temporal, half_tr_min_m=[.5, float("nan")]),
        ]
        for index, evaluation in enumerate(cases):
            with self.subTest(case=index):
                self.assert_unclassified(
                    evaluation, "unavailable (missing or inconsistent range results)")

    def test_mixed_overview_only_draws_applicable_range_bars(self):
        scene = self.scene(self.evaluation(motion="no_person"), self.evaluation())
        self.assertEqual(len(scene.axes[0].patches), 2)
        self.assertEqual(len(scene.axes[1].patches), 1)
        self.assertIn("Not applicable",
                      [item.get_text() for item in scene.axes[1].texts])


if __name__ == "__main__":
    unittest.main()

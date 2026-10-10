"""Test the output-only conversion from tr in metres to G in radians.

MetricUnitTests checks the coefficient, threshold, score preservation and
conversion of the saved ground-truth score summary.
"""
import unittest
import numpy as np

from cornersense.core import pipeline
from cornersense.metric_units import (
    G_THRESHOLD_RAD, PHASE_SCALE_RAD_PER_M, metric_units, phase_evaluation, to_paper_g)


class MetricUnitTests(unittest.TestCase):
    def test_phase_conversion_uses_mainline_spacing(self):
        tr = np.array([0., .5, .85, 1., 2.])
        expected = tr * (2 * np.pi * pipeline.DF / pipeline.C)
        np.testing.assert_array_equal(to_paper_g(tr), expected)
        np.testing.assert_allclose(to_paper_g(tr) / PHASE_SCALE_RAD_PER_M, tr, atol=1e-15)
        self.assertEqual(metric_units()['phase_spacing_hz'], 625000.)

    def test_range_threshold_has_the_same_decision(self):
        tr = np.array([.1, .849, .85, .851, 2.])
        np.testing.assert_array_equal(tr > .85, to_paper_g(tr) > G_THRESHOLD_RAD)
        self.assertAlmostEqual(G_THRESHOLD_RAD, .01113417667911831, places=15)

    def test_reporting_preserves_metre_scores(self):
        original = dict(Ig=-.12, tr_min_m=.73, half_tr_min_m=[1.1, .73],
                        predicted_motion='moving_away', predicted_regions=['R1', 'R2'],
                        both_correct=True)
        reported = phase_evaluation(original)
        self.assertEqual({key: reported[key] for key in original}, original)
        self.assertNotIn('G_min_rad', original)
        self.assertEqual(reported['G_min_rad'], float(to_paper_g(.73)))
        self.assertEqual(reported['half_G_min_rad'], to_paper_g([1.1, .73]).tolist())

    def test_truth_summary_and_empty_halves(self):
        truth = dict(Ig=.15, tr_min_m=1., half_tr_min_m=[])
        reported = phase_evaluation(dict(tr_min_m=.98, truth_on_metric_support=truth))
        self.assertEqual(reported['truth_on_metric_support']['half_G_min_rad'], [])
        self.assertEqual(reported['truth_on_metric_support']['Ig'], .15)
        self.assertNotIn('G_min_rad', truth)


if __name__ == '__main__':
    unittest.main()

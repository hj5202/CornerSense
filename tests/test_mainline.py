"""Test PCA, power residuals, distance slope and motion compensation.

Checks cover shared numerical operations and empty candidate handling.
"""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np

from cornersense.core import pipeline
from cornersense.core.adjacent_metric import weighted_adjacent_slope


class MainlineTests(unittest.TestCase):
    @staticmethod
    def synthetic_recording():
        fs = 100.
        times = np.arange(600) / fs
        phase = 2 * np.pi * 1.5 * times
        csi = 1 + .08 * np.exp(1j * (phase[:, None] + np.linspace(.1, .4, 30)))
        return fs, times, np.repeat(csi[:, None, :], 3, axis=1)

    def test_pca_score_and_loading(self):
        t = np.linspace(0, 4, 401)
        x = np.column_stack([np.cos(2*np.pi*t), 2*np.sin(2*np.pi*t), t])
        score, loading, singular = pipeline.stage_pca(x)
        u, s, vt = np.linalg.svd(x-x.mean(axis=0), full_matrices=False)
        np.testing.assert_array_equal(score, (u[:, :2]*s[:2])[:, 0])
        np.testing.assert_array_equal(loading, vt[0])
        np.testing.assert_array_equal(singular, s)

    def test_power_residual_removes_fitted_component(self):
        fs = 100.
        t = np.arange(600)/fs
        score = np.cos(2*np.pi*1.5*t)
        phase, amplitude = pipeline.score_phase(score, 1.5, fs)
        basis = np.column_stack([amplitude*np.cos(phase), amplitude*np.sin(phase)])
        power = basis @ np.array([[1., 2., 3.], [-2., 1., .5]])
        residual = pipeline.deflate_beat_ls(power, score, 1.5, fs)
        np.testing.assert_allclose(residual, 0., atol=1e-12)

    def test_adjacent_slope(self):
        f = pipeline.KG*pipeline.DF
        distance = 1.7
        phase = .3 + 2*np.pi*f*distance/pipeline.C
        fit = weighted_adjacent_slope(f, phase, np.linspace(.2, 1., len(f)))
        self.assertAlmostEqual(fit['slope']*pipeline.C/(2*np.pi), distance, places=12)

    def test_robot_compensation_uses_timestamps(self):
        times = np.array([3., 3.1, 3.35, 3.8])
        actual = pipeline.rmpoc_correction(times, .2, .9, .6, flip=-1.)
        np.testing.assert_allclose(actual, -.2*(.9-.6)*(times-times[0]), atol=0.)

    def test_conjugate_distance_check(self):
        first = dict(D=1.3, f=2.)
        self.assertEqual(pipeline.cross_pair_distance_check(first, dict(D=-1.3, f=-2.)), 'OK')
        self.assertEqual(pipeline.cross_pair_distance_check(first, dict(D=-1.3, f=2.)), 'MISMATCH')

    def test_pca_match_without_frequency_support(self):
        score = np.ones(100)
        tracked = [dict(f=1000., amp=score, Phi=np.zeros(100))]
        self.assertEqual(pipeline.pca_beat_match(score, tracked, 100., 100),
                         (None, 0.0, 0.0, 1))

    def test_stage1_without_candidates_is_unconfirmed(self):
        fs, times, csi = self.synthetic_recording()
        with patch.object(pipeline, 'beat_census', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            result = pipeline._process_one(
                csi, fs, times, 'empty-candidates', 0, len(times),
                dict(th_r=.6, th_u=.4), None)
        self.assertEqual(result, dict(name='empty-candidates', status='stage1-unconfirmed'))

    def test_stage2_without_match_preserves_stage1_result(self):
        fs, times, csi = self.synthetic_recording()
        key = dict(f=1.5, pair=(0, 1), prom=100.)
        tracked = (2 * np.pi * 1.5 * times, np.ones(len(times)),
                   1.3, .1, dict(trend_ts=-.2))
        results = []
        for stage2_match in [(0, 0.0, 0.0, 1), (None, 0.0, 0.0, 1)]:
            log = io.StringIO()
            with patch.object(pipeline, 'beat_census', return_value=[key]), \
                    patch.object(pipeline, 'track_key', return_value=tracked), \
                    patch.object(pipeline, 'pca_beat_match',
                                 side_effect=[(0, .95, 0.0, 1), stage2_match]), \
                    contextlib.redirect_stdout(log):
                result = pipeline._process_one(
                    csi, fs, times, 'stage1-candidate', 0, len(times),
                    dict(th_r=.6, th_u=.4), None)
            self.assertEqual(result['src_stage'], 1)
            self.assertIn('tr(t):', log.getvalue())
            self.assertIn('[q5 ', log.getvalue())
            self.assertNotIn('R1=', log.getvalue())
            results.append(result)
        for field in ['Dphys', 'tr', 'metric_t']:
            np.testing.assert_array_equal(results[0][field], results[1][field])
        for field in ['Ig', 'Ig_end', 'Ig_signed', 'lock1', 'lock2',
                      'trend_pca', 'trend_key', 'pca_agree', 'src_stage']:
            self.assertEqual(results[0][field], results[1][field])


if __name__ == '__main__':
    unittest.main()

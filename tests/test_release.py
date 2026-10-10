"""Test the published sample layout, calibration and saved predictions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import verify_package, verify_results


class ReleaseTests(unittest.TestCase):
    def test_published_package(self):
        report = verify_package()
        self.assertEqual(report, {
            'raw_files_verified': 35,
            'calibration_records_verified': 4,
        })
        sample = ROOT / 'sample'
        recordings = [path for path in sample.rglob('*')
                      if path.is_file() and not path.suffix
                      and not any(part.startswith('.')
                                  for part in path.relative_to(sample).parts)]
        self.assertEqual(len(recordings), report['raw_files_verified'])

    def test_all_real_samples_have_consistent_results(self):
        report = verify_results(ROOT / 'sample_results/results.json')
        self.assertEqual(report['total'], 35)
        self.assertEqual(report['motion_total'], 35)
        self.assertEqual(report['motion_correct'], 34)
        self.assertEqual(report['range_total'], 34)
        self.assertEqual(report['region_correct'], 32)
        self.assertEqual(report['both_total'], 34)
        self.assertEqual(report['both_correct'], 32)
        self.assertEqual(report['noperson_total'], 1)
        self.assertEqual(report['noperson_correct'], 1)
        self.assertEqual(report['saved_curves_recomputed'], 35)


if __name__ == '__main__':
    unittest.main()

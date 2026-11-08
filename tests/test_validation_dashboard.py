"""Checks against real held-out predictions; requires local validation artifacts."""
import unittest

import numpy as np
import torch

from dashboard.backend import (CHECKPOINTS, OUT, load_heldout, load_report,
                               sha256, summarize, validate_checkpoint, window_table)


class ValidationDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.summary = load_report()

    def test_training_subjects_cannot_be_evaluated(self):
        for subjects in (['AB06'], ['AB23', 'AB06'], [], ['AB23', 'AB23']):
            with self.assertRaises(ValueError):
                load_heldout(subjects)

    def test_fresh_predictions_reproduce_saved_runs(self):
        references = {
            'original': (OUT / 'final/classifier/predictions.npz', 'predictions'),
            'adapted': (OUT / 'phone_adaptation_v1/camargo_retention_predictions.npz', 'probabilities'),
            'angles': (OUT / 'final/regressor/predictions.npz', 'predictions'),
        }
        for key, (path, field) in references.items():
            with self.subTest(model=key), np.load(path, allow_pickle=False) as reference:
                np.testing.assert_array_equal(reference['labels'], self.data['labels'])
                np.testing.assert_array_equal(reference['subjects'], self.data['subjects'])
                np.testing.assert_allclose(self.data[key], reference[field], atol=1e-4, rtol=1e-5)

    def test_report_uses_expected_test_subjects_and_unchanged_checkpoints(self):
        self.assertEqual(set(self.data['subjects']), {'AB23', 'AB24', 'AB25'})
        self.assertEqual(len(self.data['labels']), 1910)
        for key, path in CHECKPOINTS.items():
            self.assertEqual(sha256(path), self.summary['provenance']['checkpoints'][key]['sha256'])

    def test_filtered_metrics_match_direct_calculation(self):
        mask = (self.data['subjects'] == 'AB23') & (self.data['labels'] == 0)
        report = summarize(self.data, mask)
        for key in ('original', 'adapted'):
            expected = np.mean(self.data[key][mask].argmax(1) == self.data['labels'][mask])
            self.assertAlmostEqual(report[key]['accuracy'], expected)
        error = self.data['angles'][mask].astype(np.float64) - self.data['truth'][mask]
        np.testing.assert_allclose(report['angles']['rmse_degrees'], np.sqrt(np.mean(error ** 2, axis=(0, 1))))
        with self.assertRaises(ValueError):
            summarize(self.data, np.zeros(len(mask), dtype=bool))

    def test_exports_have_one_row_per_window_and_reproduce_metrics(self):
        table = window_table(self.data)
        self.assertEqual(len(table), 1910)
        self.assertTrue(table['window_id'].is_unique)
        for key in ('original', 'adapted'):
            self.assertAlmostEqual(table[key + '_correct'].mean(), self.summary[key]['accuracy'])
            np.testing.assert_allclose(self.data[key].sum(1), 1, atol=1e-6)
        rmse = table[['hip_rmse_degrees', 'knee_rmse_degrees', 'ankle_rmse_degrees']].to_numpy()
        np.testing.assert_allclose(np.sqrt(np.mean(rmse ** 2, axis=0)), self.summary['angles']['rmse_degrees'])

    def test_wrong_window_length_or_scaler_is_rejected(self):
        checkpoint = torch.load(CHECKPOINTS['original'], map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint, 'classifier')
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(checkpoint, window_length=256), 'classifier')
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(checkpoint, scale=torch.zeros(12)), 'classifier')


if __name__ == '__main__':
    unittest.main()

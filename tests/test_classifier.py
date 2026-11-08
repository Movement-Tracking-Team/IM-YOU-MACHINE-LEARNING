import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from biomat.classifier import MovementCNN, LABEL_MAP, fit_scaler, features, load_data
from biomat.classify_phone import prepare_phone_windows, COLUMNS, SENSORS


class ClassifierTests(unittest.TestCase):
    def test_network_gradient_and_shapes(self):
        model = MovementCNN()
        x = torch.randn(4, 256, 18)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 5))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 4])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_scaler_and_features(self):
        x = np.random.default_rng(0).normal(size=(8, 256, 18)).astype(np.float32)
        mean, scale = fit_scaler(x)
        np.testing.assert_allclose(((x - mean) / scale).mean(axis=(0, 1)), 0, atol=1e-6)
        self.assertEqual(features(x).shape, (8, 90))
        self.assertEqual(LABEL_MAP['levelground1'], LABEL_MAP['levelground2'])

    def test_no_padding_and_no_label_inputs(self):
        x, y, subjects = load_data(['AB06'])
        self.assertEqual(x.shape[1:], (256, 18))
        self.assertEqual(set(y), set(range(5)))
        self.assertEqual(set(subjects), {'AB06'})

    def test_phone_alignment_and_reject_bad_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, cal = {}, {'verified_against_training_conventions': True}
            for i, sensor in enumerate(SENSORS):
                t = np.arange(600) / 100 + i
                a = np.column_stack([t, np.tile(np.arange(6), (600, 1)) + 10 * i])
                path = Path(tmp) / (sensor + '.csv')
                pd.DataFrame(a, columns=COLUMNS).to_csv(path, index=False)
                paths[sensor] = path
                cal[sensor] = {'time_offset_s': -i, 'accel_scale_to_training': 1,
                               'gyro_scale_to_training': 1, 'rotation_to_training': np.eye(3).tolist()}
            x, times, rates = prepare_phone_windows(paths, cal)
            self.assertEqual(x.shape, (3, 256, 18))
            np.testing.assert_allclose(x[0, 0], np.concatenate([np.arange(6) + 10 * i for i in range(3)]))
            cal['verified_against_training_conventions'] = False
            with self.assertRaises(ValueError):
                prepare_phone_windows(paths, cal)
            cal['verified_against_training_conventions'] = True
            frame = pd.read_csv(paths['foot'])
            frame.loc[1, 'timestamp_s'] = 0
            frame.to_csv(paths['foot'], index=False)
            with self.assertRaises(ValueError):
                prepare_phone_windows(paths, cal)


if __name__ == '__main__':
    unittest.main()

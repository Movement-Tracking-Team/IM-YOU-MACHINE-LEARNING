import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from biomat import two_sensor_100hz as experiment


class TwoSensor100HzTests(unittest.TestCase):
    def test_transient_checkpoint_lock_is_retried(self):
        locked = OSError('temporary sharing violation')
        locked.winerror = 32
        source = Mock()
        source.replace.side_effect = [locked, locked, None]
        with patch.object(experiment.time, 'sleep') as sleep:
            experiment.replace_checkpoint(source, 'latest.pt')
        self.assertEqual(source.replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_preparation_retains_100hz_targets_and_has_no_60hz_resampling(self):
        rng = np.random.default_rng(28)
        x = rng.normal(size=(2, 256, 18)).astype(np.float32)
        x[:, :, 8] -= 5
        x[:, :, 14] -= 5
        y = rng.normal(size=(2, 256, 3)).astype(np.float32)
        root = experiment.ROOT / ('test_100hz_' + uuid.uuid4().hex)
        root.mkdir()
        try:
            cache, output = root / 'cache', root / 'output'
            cache.mkdir()
            np.savez(cache / 'test.npz', x=x, y=y, valid=np.array([256, 100]),
                     activity=np.array(['levelground1', 'stairascent']))
            with patch.object(experiment, 'CACHE', cache), patch.object(experiment, 'OUT', output):
                experiment.prepare(['test'])
            with np.load(output / 'prepared/test.npz') as actual:
                self.assertEqual(actual['x'].shape, (1, 250, 12))
                np.testing.assert_array_equal(actual['y'], y[:1, :250])
                expected, _ = experiment.canonicalize(sosfiltfilt(
                    butter(4, 20, fs=100, output='sos'), x[:1, :250, 6:18], axis=1))
                np.testing.assert_array_equal(actual['x'], expected)
                np.testing.assert_array_equal(actual['labels'], [0])
        finally:
            root.resolve().relative_to(experiment.ROOT.resolve())
            shutil.rmtree(root)

    def test_rotation_and_acceleration_scale_invariance_at_100hz(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(3, 250, 12))
        x[:, :, 2] -= 5
        x[:, :, 8] -= 5
        rotated = x.copy()
        for start in (0, 6):
            rotation = Rotation.random(random_state=rng).as_matrix()
            rotated[:, :, start:start+3] = x[:, :, start:start+3] @ rotation.T * 9.80665
            rotated[:, :, start+3:start+6] = x[:, :, start+3:start+6] @ rotation.T
        original, _ = experiment.canonicalize(x)
        changed, _ = experiment.canonicalize(rotated)
        np.testing.assert_allclose(original, changed, atol=1e-5)

    def test_100hz_model_forward_and_backward(self):
        torch.set_num_threads(2)
        x = torch.randn(2, 250, 12)
        for task, shape in [('classifier', (2, 5)), ('regressor', (2, 250, 3))]:
            model = experiment.new_model(task)
            prediction = model(x)
            self.assertEqual(tuple(prediction.shape), shape)
            prediction.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            experiment.canonicalize(np.zeros((1, 150, 12)))


if __name__ == '__main__':
    unittest.main()

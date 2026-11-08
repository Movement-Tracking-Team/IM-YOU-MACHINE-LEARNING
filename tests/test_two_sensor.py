import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from biomat.two_sensor import canonicalize, new_model, LENGTH, reg_score


class TwoSensorTests(unittest.TestCase):
    def test_independent_mounting_rotations_and_acceleration_units(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(3, LENGTH, 12))
        x[:, :, 2] -= 5
        x[:, :, 8] -= 5
        y = x.copy()
        for start in [0, 6]:
            rotation = Rotation.random(random_state=rng).as_matrix()
            y[:, :, start:start+3] = (x[:, :, start:start+3] @ rotation.T)*9.80665
            y[:, :, start+3:start+6] = x[:, :, start+3:start+6] @ rotation.T
        a, _ = canonicalize(x)
        b, _ = canonicalize(y)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_twelve_channel_models_and_three_angle_outputs(self):
        torch.set_num_threads(2)
        x = torch.randn(2, LENGTH, 12)
        for task, shape in [('classifier', (2, 5)), ('regressor', (2, LENGTH, 3))]:
            model = new_model(task)
            value = model(x)
            self.assertEqual(tuple(value.shape), shape)
            value.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        truth = np.zeros((2, LENGTH, 3))
        pred = truth + np.array([1, 2, 3])
        result = reg_score(truth, pred, np.array(['a','b']), np.array(['walking','walking']))
        self.assertEqual(result['rmse_degrees'], [1.,2.,3.])
        self.assertEqual(result['subject_mean_rmse_degrees'], 2.)

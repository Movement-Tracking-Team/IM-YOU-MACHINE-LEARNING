"""Scientific parity checks against the locally cloned upstream implementation."""
import ast
import math
from pathlib import Path
from typing import Optional
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from biomat.model import BioMAT
from biomat.prepare import activity_segments, downsample, windows, scalar_text
from biomat.train import metrics

UPSTREAM = Path(__file__).resolve().parents[2] / 'BioMAT-review'


class AutoModule(nn.Module):
    """Match fastai Module initialization for executing upstream class bodies."""
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj


class ReproductionTests(unittest.TestCase):
    def test_original_architecture_outputs_and_gradients(self):
        tree = ast.parse((UPSTREAM / 'model/transformer_tsai.py').read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        scope = dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Optional=Optional,
                     Module=AutoModule, Flatten=nn.Flatten, math=math,
                     ifnone=lambda a, b: b if a is None else a,
                     default_device=lambda: torch.device('cpu'))
        exec(compile(tree, 'upstream_model', 'exec'), scope)
        original = scope['TransformerTSAI'](18, 3, 16, d_model=32, n_heads=4,
                    d_ff=16, n_layers=2, res_dropout=0.5, fc_dropout=0.5).double().eval()
        port = BioMAT(d_model=32, heads=4, d_ff=16, layers=2, seq_len=16).double().eval()
        port.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 16, 18, dtype=torch.float64, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        a, b = original(x), port(xx)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(x.grad, xx.grad, atol=1e-10, rtol=1e-9)

    def test_activity_selection_matches_upstream(self):
        tree = ast.parse((UPSTREAM / 'dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_activity_based_filter']
        scope = {'np': np, 'pd': pd}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'upstream_selection', 'exec'), scope)
        original = scope['DataSet']()
        original.config = {'dataset_name': 'camargo'}
        original.selected_trial_type = ['levelground', 'ramp', 'stair']
        original.selected_activity_label = ['all_split']
        examples = {
            'levelground': ['stand', 'stand-walk', 'walk', 'turn2', 'walk', 'turn1', 'walk', 'walk-stand', 'stand'],
            'ramp': ['idle', 'walk-rampascent', 'rampascent', 'rampascent-walk', 'idle', 'walk-rampdescent', 'rampdescent', 'rampdescent-walk'],
            'stair': ['idle', 'walk-stairascent', 'stairascent', 'stairascent-walk', 'idle', 'walk-stairdescent', 'stairdescent', 'stairdescent-walk'],
        }
        for mode, labels in examples.items():
            df = pd.DataFrame({'Label': labels, 'trialType': mode})
            x = np.arange(len(labels))[:, None]
            ox, _, ol = original.run_activity_based_filter([x], [x], [df])
            expected = {ll['trialType2'].iloc[0]: xx[:, 0] for ll, xx in zip(ol, ox)}
            for activity, idx in activity_segments(labels, mode):
                np.testing.assert_array_equal(idx, expected[activity])

    def test_resampling_and_windowing_match_upstream(self):
        scope = {'np': np}
        tree = ast.parse((UPSTREAM / 'preprocessing/segmentation/fixwindowsegmentation.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        exec(compile(tree, 'upstream_windows', 'exec'), scope)
        original = scope['FixWindowSegmentation']([], [], [], 256, 0.5)
        for n in [4, 129, 255, 256, 257, 384, 513]:
            x = np.arange(n * 3, dtype=float).reshape(n, 3)
            actual = np.stack([w[0] for w in windows(x, x)])
            np.testing.assert_array_equal(actual, original.fixsize_sliding_window(x))
        x = np.arange(1000, dtype=float)[:, None]
        np.testing.assert_allclose(downsample(x)[:, 0], np.linspace(0, 999, 500))

    def test_masked_metric_and_subject_aggregation(self):
        y = np.zeros((2, 256, 3), dtype=np.float32)
        p = np.zeros_like(y)
        p[0, :128] = 2
        p[1] = 4
        result = metrics(p, y, np.array([128, 256]), np.array(['levelground1'] * 2), np.array(['A', 'B']))
        np.testing.assert_allclose(result['valid_frames_rmse_degrees'], np.sqrt(12))
        self.assertAlmostEqual(result['subject_activity_macro_rmse_degrees'], (np.sqrt(2) + 4) / 2)

    def test_matlab_cell_label(self):
        self.assertEqual(scalar_text(np.array(['walk'])), 'walk')


if __name__ == '__main__':
    unittest.main()

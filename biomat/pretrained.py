"""Run the authors' downloaded weights locally, without modifying them."""
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import BioMAT
from .prepare import DATA, ROOT, TEST
from .train import load_subjects, metrics, predict


class Snapshot(nn.Module):
    """Only holds checkpoint attributes/tensors; no downloaded forward method."""


def read_author_model(path):
    names = ['DecoderFC', 'TransformerTSAI', '_MultiHeadAttention', '_TSTEncoder', '_TSTEncoderLayer']
    allowed = [(Snapshot, 'model.transformer_tsai.' + n) for n in names]
    allowed += [(Snapshot, 'fastai.layers.Flatten'), nn.Linear, nn.Dropout,
                nn.BatchNorm1d, nn.GELU, nn.ModuleList, nn.Sequential, set]
    with torch.serialization.safe_globals(allowed):
        snapshot = torch.load(path, map_location='cpu', weights_only=True)
    if snapshot.classification or snapshot.c_out != 3 or snapshot.seq_len != 256:
        raise ValueError('Unexpected checkpoint task/shape')
    state = snapshot.state_dict()
    # The saved older model has an unused classifier in addition to regression heads.
    ignored = [k for k in state if k.startswith('classifier.')]
    assert set(ignored) == {'classifier.fc_classifier.weight', 'classifier.fc_classifier.bias'}
    model = BioMAT()
    model.load_state_dict({k: v for k, v in state.items() if k not in ignored}, strict=True)
    return model.eval(), ignored


def evaluate(model, data, transform):
    x, y, valid, activities, subjects = data
    x = transform(x)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=25)
    start = time.monotonic()
    pred = predict(model, loader, torch.device('cpu'))
    assert np.isfinite(pred).all()
    return pred, metrics(pred, y, valid, activities, subjects), time.monotonic() - start


def main():
    torch.set_num_threads(4)
    folder = DATA / 'biomat_reproduction/pretrained'
    source = folder / 'BioMAT.pt'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    model, ignored = read_author_model(source)
    development = load_subjects(['AB06'])
    with np.load(DATA / 'biomat_reproduction/runs/paper_standard_v1/final/scaler.npz') as s:
        mean, scale = s['mean'], s['scale']
    diagnostics = {}
    for mode, transform in [('raw_upstream_behavior', lambda x: x),
                             ('our_training_scaler_diagnostic', lambda x: (x - mean) / scale)]:
        _, result, elapsed = evaluate(model, development, transform)
        diagnostics[mode] = {'metrics': result, 'cpu_seconds': elapsed}
        print('AB06_DIAGNOSTIC', mode, result['pooled_mean_rmse_degrees'], flush=True)
    # Fixed raw-input behavior from released DataSetBuilder; no test-driven scaling selection.
    test = load_subjects(TEST)
    pred, result, elapsed = evaluate(model, test, lambda x: x)
    output = {'source_sha256': original_hash, 'device': 'cpu', 'source_modified': False,
              'ignored_nonregression_parameters': ignored,
              'input_policy': 'Raw features, matching released upstream DataSetBuilder behavior',
              'limitations': 'Author scaler and exact training split not bundled/verified; '
                             'uses our raw-data conversion rather than original processed pickle',
              'development_diagnostics': diagnostics, 'test_subjects': TEST,
              'test_windows': len(pred), 'test_cpu_seconds': elapsed, 'test_metrics': result}
    (folder / 'inference_results.json').write_text(json.dumps(output, indent=2))
    np.savez(folder / 'test_predictions.npz', predictions=pred, targets=test[1], valid=test[2],
             activities=test[3], subjects=test[4])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    idx = int(np.flatnonzero((test[4] == 'AB23') & (test[3] == 'levelground1') & (test[2] == 256))[0])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for j, (ax, name) in enumerate(zip(axes, ['Hip', 'Knee', 'Ankle'])):
        ax.plot(np.arange(256) / 100, test[1][idx, :, j], label='Dataset reference', color='#196da8')
        ax.plot(np.arange(256) / 100, pred[idx, :, j], '--', label='Authors pretrained model', color='#e77b24')
        ax.set_ylabel(name + ' angle (degrees)')
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel('Seconds within window')
    fig.suptitle('Authors downloaded BioMAT weights: first full AB23 walking window\nRaw input follows released repository behavior')
    fig.savefig(folder / 'pretrained_example.png', dpi=160)
    plt.close(fig)
    ours = json.loads((DATA / 'biomat_reproduction/runs/paper_standard_v1/final/metrics.json').read_text())
    lines = ['# Authors pretrained model: local inference', '',
             'Successfully loaded the downloaded weights using restricted tensor deserialization and ran CPU inference. '
             'The source file was not modified. The original model has an unused classification head; all regression '
             'parameters loaded strictly into the compatible architecture.', '',
             '## Local test-subject comparison', '',
             '| Joint | Authors weights, raw inputs | Our trained model, standardized inputs |',
             '| --- | ---: | ---: |']
    for name, a, b in zip(['Hip', 'Knee', 'Ankle'], result['pooled_rmse_degrees'], ours['pooled_rmse_degrees']):
        lines.append(f'| {name} | {a:.3f} degrees | {b:.3f} degrees |')
    lines += [f"| Mean | {result['pooled_mean_rmse_degrees']:.3f} degrees | {ours['pooled_mean_rmse_degrees']:.3f} degrees |", '',
              f'Used {len(pred)} windows from AB23/AB24/AB25. CPU inference took {elapsed:.2f} seconds '
              '(model loading and data preparation excluded).', '',
              '## Preprocessing checks', '',
              'AB06 development-subject diagnostics, not generalization scores:', '']
    for name, d in diagnostics.items():
        lines.append(f"- {name}: {d['metrics']['pooled_mean_rmse_degrees']:.3f} degrees.")
    lines += ['', 'The authors checkpoint contains no verified input scaler. Raw inputs follow the released '
              'repository behavior; our scaler diagnostic is not the authors original normalization. '
              'The exact training membership of this downloaded checkpoint is not verified, so its local test score '
              'is not proof of performance on people unseen by that checkpoint. Dataset/preprocessing differences '
              'also prevent interpreting this as an exact paper replication or a controlled model comparison.', '',
              'All predictions, diagnostic details and the example plot are in the ignored pretrained folder.', '']
    (ROOT / 'reports/pretrained_results.md').write_text('\n'.join(lines))
    print(json.dumps({'test_metrics': result, 'test_cpu_seconds': elapsed}, indent=2))


if __name__ == '__main__':
    main()

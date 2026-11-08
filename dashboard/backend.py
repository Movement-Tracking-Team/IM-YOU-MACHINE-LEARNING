"""Re-run saved weights on held-out Camargo data; never fit on evaluation data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import torch

from biomat.classifier import CLASSES, score
from biomat.prepare import ROOT, TEST, TARGETS
from biomat.two_sensor_100hz import OUT, PREPROCESS, load, new_model, predict, reg_score

REPORT = ROOT / 'reports' / 'camargo_dashboard_validation'
CHECKPOINTS = {
    'original': OUT / 'final/classifier/model.pt',
    'adapted': OUT / 'phone_adaptation_v1/final/model.pt',
    'angles': OUT / 'final/regressor/model.pt',
}
RUN_LOCK = threading.Lock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(checkpoint, task):
    expected = {'task': task, 'sample_rate': 100, 'window_length': 250,
                'sites': ['shank', 'thigh'], 'preprocess': PREPROCESS,
                'classes': CLASSES, 'targets': TARGETS}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'Incompatible checkpoint: {key} must be {value!r}')
    mean = checkpoint['mean'].numpy()
    scale = checkpoint['scale'].numpy()
    if mean.shape != (12,) or scale.shape != (12,):
        raise ValueError('Expected twelve-channel training scaler')
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid training scaler')
    return mean, scale


def load_heldout(subjects):
    subjects = list(subjects)
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(TEST):
        raise ValueError('Evaluation accepts only unique held-out subjects AB23, AB24, AB25')
    x, y, labels, activity, people = load(subjects)
    if x.shape != (len(labels), 250, 12) or y.shape != (len(labels), 250, 3):
        raise ValueError('Unexpected prepared data shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Prepared data contains nonfinite values')
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError('Unknown activity label')
    return x, y, labels, activity, people


def summarize(data, mask=None):
    mask = np.ones(len(data['labels']), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != data['labels'].shape or not mask.any():
        raise ValueError('Select at least one evaluation window')
    return {
        'original': score(data['labels'][mask], data['original'][mask], data['subjects'][mask]),
        'adapted': score(data['labels'][mask], data['adapted'][mask], data['subjects'][mask]),
        'angles': reg_score(data['truth'][mask], data['angles'][mask], data['subjects'][mask], data['activity'][mask]),
    }


def window_table(data):
    labels = np.asarray(CLASSES)
    table = pd.DataFrame({'window_id': np.arange(len(data['labels'])), 'subject': data['subjects'],
                          'activity': data['activity'], 'reference': labels[data['labels']]})
    for key in ('original', 'adapted'):
        pred = data[key].argmax(1)
        table[key + '_prediction'] = labels[pred]
        table[key + '_correct'] = pred == data['labels']
        table[key + '_probability'] = data[key].max(1)
    errors = np.sqrt(np.mean((data['angles'].astype(np.float64) - data['truth']) ** 2, axis=1))
    for j, name in enumerate(('hip', 'knee', 'ankle')):
        table[name + '_rmse_degrees'] = errors[:, j]
    return table


def evaluate(device='auto', progress=None):
    """Fresh inference from checkpoints, writing a reproducible local report."""
    if not RUN_LOCK.acquire(blocking=False):
        raise RuntimeError('A validation run is already in progress')
    try:
        return _evaluate(device, progress or (lambda message: print(message, flush=True)))
    finally:
        RUN_LOCK.release()


def _evaluate(device, progress):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress('Loading held-out Camargo subjects AB23, AB24, AB25')
    x, y, labels, activity, subjects = load_heldout(TEST)
    data = dict(truth=y, labels=labels, activity=activity, subjects=subjects)
    sources = {}
    for key, path in CHECKPOINTS.items():
        progress(f'Running {key} model on {len(x):,} windows ({device})')
        sources[key] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        task = 'regressor' if key == 'angles' else 'classifier'
        mean, scale = validate_checkpoint(checkpoint, task)
        model = new_model(task).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        data[key] = predict(model, x, mean, scale, device, task)
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite output from {key}')
        del model, checkpoint
        if device == 'cuda':
            torch.cuda.empty_cache()
    summary = summarize(data)
    summary['provenance'] = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'device': device,
        'torch_version': torch.__version__, 'subjects': TEST, 'windows': len(x),
        'sample_rate_hz': 100, 'window_samples': 250, 'input_channels': 12,
        'preprocessing': PREPROCESS, 'checkpoints': sources,
        'prepared_data': {s: sha256(OUT / 'prepared' / (s + '.npz')) for s in TEST},
        'interpretation': 'Previously used subject-disjoint test set, not a new independent cohort. '
                          'Overlapping windows are correlated. No training or scaler fitting in this run. '
                          'Adapted classifier includes phone training; BioMAT is unchanged. '
                          'Angles are evaluated against Camargo inverse-kinematics references in degrees.',
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORT / 'predictions.npz', **data)
    window_table(data).to_csv(REPORT / 'window_metrics.csv', index=False)
    summary['provenance']['predictions_sha256'] = sha256(REPORT / 'predictions.npz')
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    progress('Validation complete')
    return data, summary


def load_report():
    summary = json.loads((REPORT / 'summary.json').read_text(encoding='utf-8'))
    if sha256(REPORT / 'predictions.npz') != summary['provenance']['predictions_sha256']:
        raise ValueError('Saved predictions do not match the report; rerun validation')
    with np.load(REPORT / 'predictions.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    return data, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    _, result = evaluate(args.device)
    print(json.dumps({k: result[k]['accuracy'] for k in ('original', 'adapted')}, indent=2))
    print('Joint RMSE (degrees):', result['angles']['rmse_degrees'])

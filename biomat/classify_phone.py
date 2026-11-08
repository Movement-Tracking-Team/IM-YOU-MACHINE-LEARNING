"""Offline inference for three phone CSVs after verified sensor calibration."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

from .classifier import CLASSES, OUT, MovementCNN, cnn_probabilities, features

SENSORS = ['foot', 'shank', 'thigh']
COLUMNS = ['timestamp_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']


def read_sensor(path, config):
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    a = df[COLUMNS].to_numpy(dtype=np.float64)
    if len(a) < 256 or not np.isfinite(a).all():
        raise ValueError(f'{path}: need at least 256 finite samples')
    t, values = a[:, 0], a[:, 1:]
    dt = np.diff(t)
    if (dt <= 0).any() or dt.max() > 0.05:
        raise ValueError(f'{path}: timestamps must increase without gaps over 50 ms')
    rate = 1 / np.median(dt)
    if rate < 95:
        raise ValueError(f'{path}: observed {rate:.1f} Hz; record at 100 Hz or higher')
    required = ['time_offset_s', 'accel_scale_to_training', 'gyro_scale_to_training', 'rotation_to_training']
    if any(config.get(k) is None for k in required):
        raise ValueError(f'{path}: calibration factors, rotation and clock offset must be supplied')
    rotation = np.asarray(config['rotation_to_training'], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('rotation_to_training must be a proper orthonormal 3x3 rotation')
    for key in ['time_offset_s', 'accel_scale_to_training', 'gyro_scale_to_training']:
        if not np.isfinite(float(config[key])):
            raise ValueError(f'Nonfinite calibration value: {key}')
    if config['accel_scale_to_training'] <= 0 or config['gyro_scale_to_training'] <= 0:
        raise ValueError('Unit conversion scale factors must be positive')
    # CSV values must already have bias/gravity handling matched to training.
    values[:, :3] = (values[:, :3] @ rotation.T) * config['accel_scale_to_training']
    values[:, 3:] = (values[:, 3:] @ rotation.T) * config['gyro_scale_to_training']
    if rate > 110:
        if np.max(np.abs(dt - np.median(dt))) > 0.1 * np.median(dt):
            raise ValueError('High-rate stream is too irregular for this resampler')
        values = sosfiltfilt(butter(4, 40, fs=rate, output='sos'), values, axis=0)
    return t + float(config['time_offset_s']), values, rate


def prepare_phone_windows(paths, calibration):
    if calibration.get('verified_against_training_conventions') is not True:
        raise ValueError('Verify units, axes, gravity handling and synchronization before phone inference; see template')
    loaded = [read_sensor(paths[s], calibration[s]) for s in SENSORS]
    start, stop = max(t[0] for t, _, _ in loaded), min(t[-1] for t, _, _ in loaded)
    if stop - start < 2.55:
        raise ValueError('Three recordings need at least 2.56 seconds of synchronized overlap')
    n = int(np.floor((stop - start) * 100 + 1e-6)) + 1
    timeline = start + np.arange(n) / 100
    signals = np.concatenate([np.column_stack([np.interp(timeline, t, v[:, j]) for j in range(6)])
                              for t, v, _ in loaded], axis=1).astype(np.float32)
    starts = np.arange(0, len(signals) - 256 + 1, 128)
    windows = np.stack([signals[i:i + 256] for i in starts])
    return windows, timeline[starts], {s: r for s, (_, _, r) in zip(SENSORS, loaded)}


def main():
    parser = argparse.ArgumentParser()
    for sensor in SENSORS:
        parser.add_argument('--' + sensor, type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT / 'phone_predictions.csv')
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    x, times, rates = prepare_phone_windows({s: getattr(args, s) for s in SENSORS}, calibration)
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    if selected == 'cnn':
        checkpoint = torch.load(OUT / 'final/cnn.pt', map_location='cpu', weights_only=True)
        model = MovementCNN()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        torch.set_num_threads(4)
        probs = cnn_probabilities(model, x, checkpoint['mean'].numpy(), checkpoint['scale'].numpy(), torch.device('cpu'))
    else:
        checkpoint = joblib.load(OUT / 'final/extra_trees.joblib')
        probs = checkpoint['model'].predict_proba(features(x))
    result = pd.DataFrame({'start_time_s': times, 'end_time_s': times + 2.55,
                           'predicted_activity': [CLASSES[i] for i in probs.argmax(1)],
                           'model_score_uncalibrated': probs.max(1)})
    for i, label in enumerate(CLASSES):
        result['score_' + label] = probs[:, i]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix('.metadata.json').write_text(json.dumps({
        'model': selected, 'observed_hz': rates, 'windows': len(x),
        'phone_accuracy_validated': False, 'calibration': calibration,
        'warning': 'Closed five-class classifier; no validated unknown rejection, running or idle recognition.'}, indent=2))
    print(result.to_string(index=False))
    print('Saved', args.output, '; phone accuracy is not yet validated.')


if __name__ == '__main__':
    main()

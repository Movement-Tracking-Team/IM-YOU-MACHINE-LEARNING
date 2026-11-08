"""Audit collected CSVs against Camargo checkpoints without fabricating missing sensors."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    session = args.session.resolve()
    out = session / 'analysis/camargo_compatibility'
    out.mkdir(parents=True, exist_ok=True)
    records, frames = {}, {}
    for phone, site in [('phone_1', 'thigh'), ('phone_2', 'shank')]:
        path = session / (phone + '.csv')
        df = pd.read_csv(path)
        frames[site] = df
        t = df.common_timestamp.to_numpy()
        dt = np.diff(t)
        values = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].to_numpy()
        records[site] = {
            'file': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'samples': len(df), 'duration_s': float(t[-1] - t[0]),
            'mean_hz': float((len(t) - 1) / (t[-1] - t[0])), 'max_gap_ms': float(dt.max() * 1000),
            'non_increasing_times': int((dt <= 0).sum()), 'nonfinite_measurements': int((~np.isfinite(values)).sum()),
            'acceleration_units': df.accel_units.unique().tolist(), 'gyro_units': df.gyro_units.unique().tolist(),
            'sync_uncertainty_ms_max': float(df.sync_uncertainty_s.max() * 1000)}
    data = root / 'Camargo et al. Dataset'
    cnn = torch.load(data / 'movement_classifier/v1/final/cnn.pt', map_location='cpu', weights_only=True)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        reg = torch.load(data / 'biomat_reproduction/runs/paper_standard_v1/final/model.pt',
                         map_location='cpu', weights_only=True)
    models = {
        'movement_classifier': {'input_channels': int(cnn['state_dict']['encoder.0.weight'].shape[1]),
                                'classes': cnn['classes'], 'sample_rate_hz': cnn['sample_rate']},
        'joint_angles': {'input_channels': int(reg['model']['W_P.weight'].shape[1]),
                         'outputs': ['hip', 'knee', 'ankle'], 'sample_rate_hz': 100}}
    assert all(m['input_channels'] == 18 for m in models.values())
    result = {'session': session.name, 'devices': records, 'models': models, 'available_channels': 12,
              'direct_inference_supported': False, 'predictions_generated': False,
              'reasons': ['Missing foot sensor (six required input channels).',
                          'Approximately 60 Hz recording versus 100 Hz training.',
                          'Mounting rotation, leg side, and Camargo unit/gravity conventions not verified.'],
              'joint_angle_ground_truth_available': False}
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for site, df in frames.items():
        axes[0].plot(df.session_time_s, np.linalg.norm(df[['ax', 'ay', 'az']], axis=1), label=site, linewidth=1)
        axes[1].plot(df.session_time_s, np.linalg.norm(df[['gx', 'gy', 'gz']], axis=1), label=site, linewidth=1)
    axes[0].set(ylabel='Acceleration magnitude (m/s^2)', title='Collected phone motion: not model predictions')
    axes[1].set(ylabel='Angular speed magnitude (deg/s)', xlabel='Session time (s)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(out / 'phone_signals.png', dpi=150)
    plt.close(fig)
    lines = ['# Camargo model compatibility with the collected two-phone trial', '',
             '**Current saved models cannot directly infer from this recording. No activity or joint-angle predictions were generated.**', '',
             '| Site | Samples | Mean Hz | Largest gap (ms) | Nonfinite values |',
             '| --- | ---: | ---: | ---: | ---: |']
    for site, r in records.items():
        lines.append(f"| {site} | {r['samples']} | {r['mean_hz']:.2f} | {r['max_gap_ms']:.2f} | {r['nonfinite_measurements']} |")
    lines += ['', 'Both saved checkpoints require 18 channels from foot, shank and thigh. These files provide 12 channels from thigh and shank. The trained time scale is 100 Hz; these recordings are approximately 60 Hz. Upsampling cannot restore missing bandwidth. The existing phone adapter requires at least 95 Hz and verified calibration.', '',
              'Phone exports contain acceleration including gravity in m/s^2 and angular velocity in deg/s. Mounting transforms and equivalence to Camargo axes and units are not established. Receiver-estimated synchronization uncertainty is not independently measured physical synchronization error.', '',
              'The earlier walking annotation is user-supplied, not a classifier result. Signal magnitude does not establish an activity label. No reference joint-angle measurements were supplied.', '',
              'A two-sensor experiment requires retraining or adapting models on Camargo thigh/shank channels, matching sample rate and preprocessing, then evaluating on held-out Camargo participants before attempting calibrated phone inference. A knee-focused regression model is a reasonable experiment; its accuracy is not established. Filling or duplicating foot channels does not make the existing models equivalent to two-sensor models.', '',
              '[Recorded signal visualization](phone_signals.png) | [Audit and checkpoint dimensions](audit.json)', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()

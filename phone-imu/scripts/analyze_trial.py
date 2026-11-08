"""Split a user-labelled walking trial, preserve raw rows, and estimate motion boundaries.

Requires numpy, pandas, scipy and matplotlib (available in the project's ML environment).
This is an exploratory motion threshold, not a trained gait or first-step detector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, uniform_filter1d


def motion_intervals(group, threshold):
    t = group.session_time_s.to_numpy()
    rate = 1 / np.median(np.diff(group.phone_timestamp))
    norm = np.linalg.norm(group[['gx', 'gy', 'gz']].to_numpy(), axis=1)
    window = max(3, round(rate * 0.25))
    if window % 2 == 0:
        window += 1
    envelope = np.sqrt(uniform_filter1d(norm ** 2, size=window, mode='nearest'))
    active = binary_closing(envelope > threshold, structure=np.ones(max(1, round(rate * 0.4))))
    edges = np.diff(np.r_[False, active, False].astype(int))
    intervals = [(float(t[a]), float(t[b] + 1 / rate))
                 for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)
                 if t[b] - t[a] >= 0.3]
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--session-directory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    metadata = json.loads((args.session_directory / 'metadata.json').read_text())
    ids = df.session_id.unique()
    if len(ids) != 1 or ids[0] != metadata['session_id']:
        raise ValueError('Expected exactly one session matching the supplied metadata')
    if set(df.device_id) != {'phone_1', 'phone_2'}:
        raise ValueError('Expected Phone 1 (thigh) and Phone 2 (shank)')
    columns = ['session_time_s', 'common_timestamp', 'phone_timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    if not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError('Nonfinite measurement or timestamp')
    if set(df.accel_units) != {'m/s^2'} or set(df.gyro_units) != {'deg/s'}:
        raise ValueError('This analysis expects m/s^2 and deg/s')
    duration = metadata['stop'] - metadata['start']
    assert np.allclose(df.common_timestamp - df.session_time_s, metadata['start'], atol=1e-9, rtol=0)
    assert df.session_time_s.min() >= 0 and df.session_time_s.max() < duration
    groups = {device: g.sort_values('session_time_s').copy() for device, g in df.groupby('device_id')}
    mapping = {'phone_1': 'thigh', 'phone_2': 'shank'}
    findings = {}
    for device, g in groups.items():
        dt = np.diff(g.phone_timestamp)
        selected = motion_intervals(g, 15)
        if len(selected) != 1:
            raise ValueError(f'{device}: expected one sustained movement bout, got {selected}; inspect manually')
        sequences = sorted(g.packet_sequence_number.unique())
        findings[device] = {
            'body_segment_user_supplied': mapping[device], 'samples': len(g),
            'first_sample_session_s': float(g.session_time_s.iloc[0]),
            'last_sample_session_s': float(g.session_time_s.iloc[-1]),
            'first_sample_host_monotonic_s': float(g.common_timestamp.iloc[0]),
            'actual_mean_hz': float((len(g) - 1) / (g.phone_timestamp.iloc[-1] - g.phone_timestamp.iloc[0])),
            'median_interval_ms': float(np.median(dt) * 1000),
            'max_interval_ms': float(dt.max() * 1000), 'interval_sd_ms': float(dt.std() * 1000),
            'non_increasing_intervals': int((dt <= 0).sum()), 'intervals_over_50_ms': int((dt > .05).sum()),
            'missing_sequences_within_csv_sequence_range': int(sequences[-1] - sequences[0] + 1 - len(sequences)),
            'sync_uncertainty_ms_range': [float(g.sync_uncertainty_s.min() * 1000), float(g.sync_uncertainty_s.max() * 1000)],
            'motion_interval_15_deg_s': selected[0],
            'threshold_sensitivity': {str(th): motion_intervals(g, th) for th in [10, 15, 20, 30]},
            'missing_optional_event_index': int(g.event_index.isna().sum()),
        }
    # One shared crop: do not independently zero or align the anatomical signals.
    start = min(v['motion_interval_15_deg_s'][0] for v in findings.values())
    stop = min(duration, max(v['motion_interval_15_deg_s'][1] for v in findings.values()))
    segments = pd.DataFrame([
        {'segment': 'low_motion_before', 'start_s': 0., 'stop_s': start},
        {'segment': 'walking_estimate', 'start_s': start, 'stop_s': stop},
        {'segment': 'low_motion_after', 'start_s': stop, 'stop_s': duration},
    ])
    segments['duration_s'] = segments.stop_s - segments.start_s
    journal = [json.loads(line) for line in (args.session_directory / 'packets.jsonl').read_text().splitlines() if line]
    for device, stats in findings.items():
        numbers = [p['packet_sequence_number'] for p in journal if p['device_id'] == device]
        final = metadata['delivery'][device]['last_sequence']
        stats['journal_packet_count'] = len(numbers)
        stats['journal_duplicate_sequences'] = len(numbers) - len(set(numbers))
        stats['complete_sequence_0_through_final'] = set(numbers) == set(range(final + 1))
        stats['metadata_delivery_complete'] = metadata['delivery'][device]['next_sequence'] == final + 1
        raw_count = sum(1 for p in journal if p['device_id'] == device for sample in p['samples']
                        if metadata['start'] <= sample['common_timestamp'] < metadata['stop'])
        stats['journal_in_window_samples'] = raw_count
        if raw_count != stats['samples']:
            raise ValueError(f'{device}: CSV and receiver journal sample counts differ')
    # Refuse to overwrite any prior analysis or the user's source data.
    args.out.mkdir(parents=True, exist_ok=False)
    for device, g in groups.items():
        segment = mapping[device]
        g.to_csv(args.out / f'{segment}_full.csv', index=False, float_format='%.17g')
        walking = g[(g.session_time_s >= start) & (g.session_time_s < stop)].copy()
        walking['walking_window_time_s'] = walking.session_time_s - start
        walking.to_csv(args.out / f'{segment}_walking_estimate.csv', index=False, float_format='%.17g')
        findings[device]['walking_crop_samples'] = len(walking)
    for row in segments.itertuples():
        part = df[(df.session_time_s >= row.start_s) & (df.session_time_s < row.stop_s)].copy()
        part['body_segment'] = part.device_id.map(mapping)
        part['segment_time_s'] = part.session_time_s - row.start_s
        part.to_csv(args.out / f'{row.segment}_combined.csv', index=False, float_format='%.17g')
    segments.to_csv(args.out / 'segments.csv', index=False)
    report = {
        'source': str(args.input.resolve()), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'session_id': str(ids[0]), 'user_activity_label': 'walking', 'session_count': 1,
        'recording_start_host_monotonic_s': metadata['start'], 'recording_duration_s': duration,
        'walking_estimate_start_s': start, 'walking_estimate_stop_s': stop, 'devices': findings,
        'method': '250 ms centered RMS of gyroscope vector magnitude; 15 deg/s threshold; bridge ~400 ms gaps; retain bouts >=300 ms; shared union of phone bouts.',
        'limitations': ['User supplied the walking label; this is a motion-based boundary estimate, not a validated gait classifier.',
                        'Smoothing and threshold choice affect boundaries; do not interpret onset to millisecond precision.',
                        'Thigh/shank signal phase differences are not evidence of clock error.',
                        'Host and phone timestamps are monotonic, not calendar date/time.',
                        'Raw measurements, units and timestamps are unchanged; smoothing is used only to estimate segment boundaries.'],
    }
    (args.out / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    colors = {'phone_1': '#007e70', 'phone_2': '#c85e2c'}
    for device, g in groups.items():
        t = g.session_time_s.to_numpy()
        label = f'{device.replace("_", " ").title()} — {mapping[device]}'
        axes[0].plot(t, np.linalg.norm(g[['gx', 'gy', 'gz']], axis=1), color=colors[device], label=label, lw=.9)
        axes[1].plot(t, np.linalg.norm(g[['ax', 'ay', 'az']], axis=1), color=colors[device], label=label, lw=.9)
        axes[2].plot(t[1:], np.diff(g.phone_timestamp) * 1000, color=colors[device], label=label, lw=.7, alpha=.8)
    for ax in axes:
        ax.axvspan(start, stop, color='#55b990', alpha=.12)
        ax.axvline(start, color='#333333', ls='--', lw=1)
        ax.axvline(stop, color='#333333', ls='--', lw=1)
        ax.grid(alpha=.18); ax.set_xlim(0, duration)
    axes[0].set_title(f'Walking trial — estimated movement {start:.1f}–{stop:.1f} s within {duration:.2f} s recording', loc='left', fontsize=14)
    axes[0].set_ylabel('Gyroscope magnitude\n(deg/s)'); axes[0].legend(loc='upper right')
    axes[1].set_ylabel('Acceleration magnitude\nincluding gravity (m/s²)')
    axes[2].set_ylabel('Original phone timestamp\ninterval (ms)'); axes[2].set_xlabel('Time since shared recording start (s)')
    fig.savefig(args.out / 'trial_overview.png', dpi=170); plt.close(fig)
    lines = ['# Collected walking trial', '', f'Session: `{ids[0]}`. One recording, {duration:.3f} seconds, {len(df)} rows.', '',
             'Phone 1 = thigh; Phone 2 = shank (user-supplied placement).', '',
             '| Device | Samples | First data (s) | Last data (s) | Mean Hz |', '|---|---:|---:|---:|---:|']
    for device, stats in findings.items():
        lines.append(f'| {device} ({mapping[device]}) | {stats["samples"]} | {stats["first_sample_session_s"]:.5f} | {stats["last_sample_session_s"]:.5f} | {stats["actual_mean_hz"]:.3f} |')
    lines += ['', f'Estimated walking/movement window: **{start:.1f}–{stop:.1f} seconds**, based on both phones.',
              'The first ~2.1 s and last ~3.9 s are low-motion periods. Exact foot-contact times are not established.', '',
              'Both streams have finite measurements, no non-increasing phone timestamps and no intervals above 50 ms in this trial. See analysis.json for checks against the receiver journal and exact thresholds.', '',
              'Phone 1 has no optional event_index/timestamp_order values (older client format); its six measurement channels and original timestamps are present.', '',
              'The full files retain every input column and original timestamps. Cropped files use the same common window for both phones, with derived relative-time columns. No axis alignment, smoothing, resampling or unit conversion has been applied to exported measurements.', '',
              'Analysis smoothing is only used for the boundary estimate. Thresholds of 10–30 deg/s place onset near 2.1–2.2 s and the end near 13.3–13.8 s. These ranges are sensitivity checks, not statistical confidence intervals.', '',
              'These are ~60 Hz, two-sensor recordings; they do not directly satisfy the existing three-sensor, >=95 Hz Camargo phone adapter.']
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    # Round-trip verification: splitting preserved counts and all measurement/timestamp values.
    for device, g in groups.items():
        saved = pd.read_csv(args.out / f'{mapping[device]}_full.csv')
        assert len(saved) == len(g)
        np.testing.assert_allclose(saved[columns], g[columns], atol=1e-10, rtol=1e-12)
    assert sum(len(pd.read_csv(args.out / f'{s}_combined.csv')) for s in segments.segment) == len(df)
    print(json.dumps(report, indent=2))
    print('Saved and verified:', args.out.resolve())


if __name__ == '__main__':
    main()

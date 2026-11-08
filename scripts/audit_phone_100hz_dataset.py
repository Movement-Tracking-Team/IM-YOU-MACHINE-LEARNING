"""Read-only audit of Sensor Logger archives; never fits or evaluates model weights."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomat.two_sensor_100hz import canonicalize

SENSORS = ['AccelerometerUncalibrated', 'GyroscopeUncalibrated', 'Accelerometer',
           'Gyroscope', 'Gravity', 'Orientation']
ACTIVITIES = ['walking', 'stair_ascent', 'stair_descent']
UNIT_SOURCE = 'https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md'
TIME_SOURCE = 'https://github.com/tszheichoi/awesome-sensor-logger#understanding-timestamps'


def labels(name):
    normalized = name.lower().replace('wailking', 'walking').replace('desctn', 'descent').replace('phoje', 'phone').replace('trail_', 'trial_').replace('trials_', 'trial_')
    activity = next(a for a in ACTIVITIES if normalized.startswith(a + '_'))
    return activity, int(re.search(r'trial_(\d+)', normalized)[1]), int(re.search(r'phone_(\d+)', normalized)[1])


def sensor_stats(df, epoch_ms):
    # Preserve integer epoch nanoseconds before differencing to avoid float loss.
    timestamps = df['time'].to_numpy(dtype=np.int64)
    dt = np.diff(timestamps).astype(np.float64) / 1e9
    duration = (int(timestamps[-1]) - int(timestamps[0])) / 1e9
    columns = ['x', 'y', 'z'] if 'x' in df else ['yaw', 'roll', 'pitch', 'qx', 'qy', 'qz', 'qw']
    values = df[columns].to_numpy(dtype=float)
    median_dt = float(np.median(dt))
    timing_residual = (timestamps - int(epoch_ms) * 1000000) / 1e9 - df.seconds_elapsed.to_numpy()
    result = {'rows': len(df), 'duration_s': duration,
              'observed_hz': (len(df) - 1) / duration, 'median_interval_ms': median_dt * 1000,
              'min_interval_ms': float(dt.min() * 1000), 'max_interval_ms': float(dt.max() * 1000),
              'p99_interval_ms': float(np.quantile(dt, .99) * 1000),
              'interval_std_ms': float(dt.std() * 1000),
              'duplicate_timestamps': int((dt == 0).sum()), 'backward_timestamps': int((dt < 0).sum()),
              'gaps_over_20ms': int((dt > .020).sum()), 'gaps_over_50ms': int((dt > .050).sum()),
              'suspected_missing_samples_from_spacing': int(np.maximum(np.rint(dt / median_dt) - 1, 0).sum()),
              'nonfinite_values': int((~np.isfinite(values)).sum()),
              'repeated_adjacent_vectors': int(np.all(np.diff(values, axis=0) == 0, axis=1).sum()),
              'first_timestamp_ns': int(timestamps[0]), 'last_timestamp_ns': int(timestamps[-1]),
              'max_time_vs_elapsed_residual_ms': float(np.max(np.abs(timing_residual)) * 1000)}
    if 'x' in df:
        norm = np.linalg.norm(values, axis=1)
        result.update({'median_vector_magnitude': float(np.median(norm)), 'p95_vector_magnitude': float(np.quantile(norm, .95)),
                       'max_vector_magnitude': float(norm.max()),
                       'axis_min_xyz': values.min(0).tolist(), 'axis_max_xyz': values.max(0).tolist(),
                       'axis_std_xyz': values.std(0).tolist()})
    else:
        q = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
        result['max_quaternion_norm_error'] = float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1)))
    return result


def interpolate(df, timeline, origin_ns):
    time = (df.time.to_numpy(dtype=np.int64) - origin_ns) / 1e9
    return np.column_stack([np.interp(timeline, time, df[c]) for c in ['x', 'y', 'z']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent.parent / 'Our Dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/phone_100hz_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, frames, sensor_rows = [], {}, []
    device_votes = defaultdict(Counter)
    for archive in sorted(args.source.glob('*.zip')):
        activity, trial, filename_phone = labels(archive.name)
        with zipfile.ZipFile(archive) as z:
            bad_member = z.testzip()
            if bad_member:
                raise ValueError(f'Corrupt archive: {archive.name}/{bad_member}')
            metadata = pd.read_csv(io.BytesIO(z.read('Metadata.csv'))).iloc[0]
            standardisation = str(metadata['standardisation']).lower() == 'true'
            if str(metadata['platform']) != 'ios' or standardisation:
                raise ValueError('This audit requires explicit review for other platforms/unit modes')
            device = str(metadata['device id'])
            device_votes[device][filename_phone] += 1
            data = {sensor: pd.read_csv(io.BytesIO(z.read(sensor + '.csv'))) for sensor in SENSORS}
            annotation = z.read('Annotation.csv').decode('utf-8-sig').strip()
            stats = {sensor: sensor_stats(df, metadata['recording epoch time']) for sensor, df in data.items()}
            content_digest = hashlib.sha256()
            for name in sorted(z.namelist()):
                content_digest.update(name.encode())
                content_digest.update(z.read(name))
            record = {'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'content_sha256': content_digest.hexdigest(), 'activity': activity, 'trial': trial,
                      'filename_phone': filename_phone, 'device_id': device, 'device_name': str(metadata['device name']),
                      'recording_epoch_ms': int(metadata['recording epoch time']),
                      'recording_time_label': str(metadata['recording time']),
                      'sample_rate_setting_ms': str(metadata['sampleRateMs']),
                      'standardisation': standardisation, 'platform': str(metadata['platform']),
                      'annotation_present': bool(annotation), 'sensors': stats}
            raw_a, raw_w = data['AccelerometerUncalibrated'], data['GyroscopeUncalibrated']
            record['raw_accel_gyro_timestamps_identical'] = np.array_equal(raw_a.time, raw_w.time)
            raw_norm = np.linalg.norm(raw_a[['x', 'y', 'z']].to_numpy(), axis=1)
            peak_index = int(np.argmax(raw_norm))
            record['raw_acceleration_peak_elapsed_s'] = float(raw_a.seconds_elapsed.iloc[peak_index])
            record['raw_acceleration_samples_above_5g'] = int((raw_norm > 5).sum())
            # Cross-check raw g data against the independently exported acceleration+gravity.
            acc, gravity = data['Accelerometer'], data['Gravity']
            if not np.array_equal(acc.time, gravity.time):
                raise ValueError('Processed acceleration/gravity timestamps need explicit alignment')
            origin = int(raw_a.time.iloc[0])
            valid = (acc.time >= raw_a.time.iloc[0]) & (acc.time <= raw_a.time.iloc[-1])
            timeline = (acc.time[valid].to_numpy(dtype=np.int64) - origin) / 1e9
            total = acc.loc[valid, ['x', 'y', 'z']].to_numpy() + gravity.loc[valid, ['x', 'y', 'z']].to_numpy()
            reconstructed = interpolate(raw_a, timeline, origin) * 9.80665
            record['raw_vs_accel_plus_gravity_rmse_m_s2'] = np.sqrt(np.mean((total - reconstructed) ** 2, axis=0)).tolist()
            for sensor, details in stats.items():
                sensor_rows.append({'file': archive.name, 'activity': activity, 'trial': trial,
                                    'sensor': sensor, **{k: v for k, v in details.items() if not isinstance(v, list)}})
            records.append(record)
            frames[archive.name] = data

    # Infer physical device roles from consistent labels across all recordings.
    # Do not rename or edit source files; record the one contradictory filename.
    roles = {}
    for device, votes in device_votes.items():
        role, count = votes.most_common(1)[0]
        if count <= sum(votes.values()) / 2:
            raise ValueError('Ambiguous physical device mapping')
        roles[device] = role
    if set(roles.values()) != {1, 2}:
        raise ValueError('Expected exactly two consistent phone roles')
    groups = defaultdict(dict)
    for record in records:
        phone = roles[record['device_id']]
        record.update({'resolved_phone': phone, 'placement': 'right_thigh' if phone == 1 else 'right_shank',
                       'filename_phone_mismatch': phone != record['filename_phone']})
        key = (record['activity'], record['trial'])
        if phone in groups[key]:
            raise ValueError('Multiple recordings for the same physical phone and trial')
        groups[key][phone] = record

    pairs, motion = [], {}
    for activity in ACTIVITIES:
        for trial in [1, 2, 3]:
            group = groups.get((activity, trial), {})
            missing = [p for p in (1, 2) if p not in group]
            pair = {'activity': activity, 'trial': trial, 'missing_phones': missing,
                    'status': 'missing_phone' if missing else 'paired_on_epoch_timestamps_unverified_clock_sync'}
            if missing:
                pairs.append(pair)
                continue
            raw = {p: {s: frames[group[p]['file']][s] for s in ['AccelerometerUncalibrated', 'GyroscopeUncalibrated']} for p in (1, 2)}
            start = max(int(df.time.iloc[0]) for sensors in raw.values() for df in sensors.values())
            stop = min(int(df.time.iloc[-1]) for sensors in raw.values() for df in sensors.values())
            duration = (stop - start) / 1e9
            timeline = np.arange(int(np.floor(duration * 100)) + 1) / 100
            signals = np.concatenate([np.column_stack([interpolate(raw[p]['AccelerometerUncalibrated'], timeline, start),
                                                      interpolate(raw[p]['GyroscopeUncalibrated'], timeline, start)]) for p in [2, 1]], axis=1)
            filtered = sosfiltfilt(butter(4, 20, fs=100, output='sos'), signals, axis=0)
            starts = np.arange(0, len(filtered) - 250 + 1, 50)
            windows = np.stack([filtered[s:s+250] for s in starts])
            _, quality = canonicalize(windows)
            accepted = np.ones(len(windows), dtype=bool)
            motion_ok = np.ones(len(windows), dtype=bool)
            axis_ok = np.ones(len(windows), dtype=bool)
            moving_fractions = []
            for site in range(2):
                fraction = (np.linalg.norm(windows[:, :, site*6+3:site*6+6], axis=2) > .25).mean(1)
                accepted &= (fraction >= .75) & (quality[site]['principal_axis_fraction'] >= .6)
                motion_ok &= fraction >= .75
                axis_ok &= quality[site]['principal_axis_fraction'] >= .6
                moving_fractions.append(fraction)
            difference_ms = (group[2]['recording_epoch_ms'] - group[1]['recording_epoch_ms'])
            pair.update({'phone_1_file': group[1]['file'], 'phone_2_file': group[2]['file'],
                         'overlap_duration_s': duration, 'phone2_minus_phone1_start_ms': difference_ms,
                         'phone1_raw_hz': group[1]['sensors']['AccelerometerUncalibrated']['observed_hz'],
                         'phone2_raw_hz': group[2]['sensors']['AccelerometerUncalibrated']['observed_hz'],
                         'aligned_samples': len(timeline), 'candidate_windows': len(windows),
                         'windows_passing_existing_gate': int(accepted.sum()),
                         'windows_failing_motion_requirement': int((~motion_ok).sum()),
                         'windows_failing_axis_requirement': int((~axis_ok).sum()),
                         'gate_coverage_fraction': float(accepted.mean()),
                         'phone1_fraction_samples_gyro_over_025': float((np.linalg.norm(filtered[:, 9:12], axis=1) > .25).mean()),
                         'phone2_fraction_samples_gyro_over_025': float((np.linalg.norm(filtered[:, 3:6], axis=1) > .25).mean()),
                         'clock_offset_estimated_or_corrected': False})
            window_table = pd.DataFrame({'start_s_from_overlap': timeline[starts], 'passes_existing_gate': accepted,
                                         'shank_moving_fraction': moving_fractions[0], 'thigh_moving_fraction': moving_fractions[1],
                                         'shank_axis_fraction': quality[0]['principal_axis_fraction'],
                                         'thigh_axis_fraction': quality[1]['principal_axis_fraction']})
            window_table.to_csv(args.output / f'{activity}_trial_{trial}_window_quality.csv', index=False)
            motion[(activity, trial)] = (timeline, filtered, accepted, starts)
            pairs.append(pair)

    all_stats = [s for r in records for s in r['sensors'].values()]
    raw_stats = [r['sensors'][s] for r in records for s in ['AccelerometerUncalibrated', 'GyroscopeUncalibrated']]
    report = {'source_directory': str(args.source.resolve()), 'archive_count': len(records),
              'archive_integrity': 'all passed CRC checks',
              'unique_archive_payloads': len({r['content_sha256'] for r in records}),
              'complete_trial_pairs': sum(not p['missing_phones'] for p in pairs),
              'missing_pairs': [p for p in pairs if p['missing_phones']],
              'filename_phone_mismatches': [r['file'] for r in records if r['filename_phone_mismatch']],
              'physical_device_roles': [{'phone': roles[d], 'device_name': next(r['device_name'] for r in records if r['device_id'] == d),
                                        'votes_by_filename_phone': dict(v)} for d, v in device_votes.items()],
              'observed_raw_hz_range': [min(s['observed_hz'] for s in raw_stats), max(s['observed_hz'] for s in raw_stats)],
              'total_nonfinite_sensor_values': sum(s['nonfinite_values'] for s in all_stats),
              'total_duplicate_sensor_timestamps': sum(s['duplicate_timestamps'] for s in all_stats),
              'total_backward_sensor_timestamps': sum(s['backward_timestamps'] for s in all_stats),
              'total_sensor_gaps_over_50ms': sum(s['gaps_over_50ms'] for s in all_stats),
              'raw_max_interval_ms': max(s['max_interval_ms'] for s in raw_stats),
              'total_candidate_windows': sum(p.get('candidate_windows', 0) for p in pairs),
              'total_windows_passing_gate': sum(p.get('windows_passing_existing_gate', 0) for p in pairs),
              'total_windows_failing_motion_requirement': sum(p.get('windows_failing_motion_requirement', 0) for p in pairs),
              'total_windows_failing_axis_requirement': sum(p.get('windows_failing_axis_requirement', 0) for p in pairs),
              'annotations_present': sum(r['annotation_present'] for r in records),
              'unit_reference': UNIT_SOURCE, 'timestamp_reference': TIME_SOURCE,
              'units': {'AccelerometerUncalibrated': 'g, includes gravity (iOS, standardisation=false)',
                        'GyroscopeUncalibrated': 'rad/s', 'Gyroscope': 'rad/s',
                        'Accelerometer': 'm/s^2, gravity removed', 'Gravity': 'm/s^2',
                        'time': 'epoch nanoseconds', 'seconds_elapsed': 'seconds since each separate recording start'},
              'placement': {'phone_1': 'right thigh', 'phone_2': 'right shank'},
              'alignment_assumption': 'Direct epoch alignment; inter-phone clock offset/drift is unverified, not inferred from recording-start differences.',
              'motion_gate_only': True, 'model_predictions_computed': False,
              'records': records, 'pairs': pairs}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pd.DataFrame(sensor_rows).to_csv(args.output / 'sensor_quality.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'sensors'} for r in records]).to_csv(args.output / 'file_manifest.csv', index=False)
    pd.DataFrame(pairs).to_csv(args.output / 'trial_pairs.csv', index=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for row, activity in enumerate(ACTIVITIES):
        for col, trial in enumerate([1, 2, 3]):
            ax = axes[row, col]
            ax.set_title(f'{activity.replace("_", " ").title()} — trial {trial}')
            if (activity, trial) not in motion:
                ax.text(.5, .5, 'Missing phone 2 recording', ha='center', va='center', transform=ax.transAxes)
            else:
                timeline, filtered, accepted, starts = motion[(activity, trial)]
                ax.plot(timeline, np.linalg.norm(filtered[:, 9:12], axis=1), label='Thigh / phone 1', lw=.9, color='#2864aa')
                ax.plot(timeline, np.linalg.norm(filtered[:, 3:6], axis=1), label='Shank / phone 2', lw=.9, color='#d97922')
                ax.axhline(.25, color='gray', lw=.7, linestyle='--')
                ax.text(.02, .95, f'{accepted.sum()}/{len(accepted)} windows pass gate', va='top', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel('Seconds from shared epoch overlap')
            ax.set_ylabel('Filtered angular speed (rad/s)')
            ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Phone recording audit — direct epoch alignment; clock synchronization unverified', fontsize=14)
    fig.savefig(args.output / 'motion_overview.png', dpi=150)
    plt.close(fig)

    lines = ['# Audit of the 100 Hz phone dataset', '',
             f"Source: `{args.source.resolve()}`", '',
             f"{len(records)} intact, distinct archives; {report['complete_trial_pairs']} complete paired trials out of 9.", '',
             'Phone 1 = right thigh; phone 2 = right shank, as confirmed by the user. No source files were changed.', '',
             '## Inventory findings', '',
             '- Walking trial 1 has phone 1 only; the phone 2 archive is missing.',
             '- `stair_ascent_phone_1_trial_2-2026-11-08_03-24-08 2.zip` is the iPhone 15 Plus (phone 2 / shank), despite its name. It is distinct from the iPhone 17 Pro Max recording with the same base name. The manifest resolves its role using device identity; the source filename was not changed.',
             '- Minor spelling differences in filenames are normalized only in the manifest.', '',
             '## Sensor quality', '',
             f"Observed raw accelerometer/gyroscope rates: {report['observed_raw_hz_range'][0]:.3f}–{report['observed_raw_hz_range'][1]:.3f} Hz.",
             f"Across all six sensor streams in all archives: {report['total_nonfinite_sensor_values']} nonfinite values; {report['total_duplicate_sensor_timestamps']} duplicate timestamps; {report['total_backward_sensor_timestamps']} backwards timestamps; {report['total_sensor_gaps_over_50ms']} gaps over 50 ms.",
             f"Maximum raw-sensor interval: {report['raw_max_interval_ms']:.4f} ms.", '',
             'The raw shank acceleration peaks reach about 8.13 g in ascent trial 1 and 9.39 g in ascent trial 2. These are brief high-amplitude events, not proof of sensor failure or clipping. Per-file peak times and counts over 5 g are recorded in audit.json; processing should retain the predefined filter rather than tune it against activity predictions.', '',
             '## Units and model input', '',
             'All exports identify iOS with Standardise Units & Frames disabled. The documented units are:', '',
             '| Stream | Units | Interpretation |', '| --- | --- | --- |',
             '| AccelerometerUncalibrated.csv | g | Raw acceleration including gravity |',
             '| GyroscopeUncalibrated.csv | rad/s | Raw angular velocity |',
             '| Gyroscope.csv | rad/s | Processed angular velocity |',
             '| Accelerometer.csv | m/s² | User acceleration, gravity removed |',
             '| Gravity.csv | m/s² | Gravity vector |',
             '| time | ns since epoch | Parse as integer; use common origin before conversion |',
             '| seconds_elapsed | seconds | Each phone recording has its own start |', '',
             f'Unit definitions: [Sensor Logger documentation]({UNIT_SOURCE}).', '',
             'The existing model expects gravity-inclusive acceleration. Use raw AccelerometerUncalibrated with GyroscopeUncalibrated for a consistent raw sensor path, with explicit g/rad/s labels. Alternatively reconstruct total acceleration from time-aligned Accelerometer + Gravity; this is a different processed sensor path and should be declared, not selected according to classifier accuracy.',
             'CSV column order is time, seconds_elapsed, z, y, x. Read axes by name (x, y, z), not by their position. Do not apply degrees-to-radians conversion to these gyro values. Gravity removal would break the intended normalization reference.', '',
             '## Trial pairing and motion quality', '',
             '| Activity | Trial | Overlap (s) | Thigh Hz | Shank Hz | Start difference, phone 2 minus phone 1 (ms) | Passing windows |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pair in pairs:
        if pair['missing_phones']:
            lines.append(f"| {pair['activity']} | {pair['trial']} | Missing phone 2 | — | — | — | — |")
        else:
            lines.append(f"| {pair['activity']} | {pair['trial']} | {pair['overlap_duration_s']:.2f} | {pair['phone1_raw_hz']:.3f} | {pair['phone2_raw_hz']:.3f} | {pair['phone2_minus_phone1_start_ms']} | {pair['windows_passing_existing_gate']}/{pair['candidate_windows']} |")
    lines += ['', 'Start differences describe when recording began; they are NOT estimates of clock error. Using separate seconds_elapsed=0 as if both phones started simultaneously would misalign the streams.',
              'The pair diagnostics use direct epoch timestamps without time-shift optimization or model predictions. Two devices sharing epoch units does not establish synchronization accuracy. No exported synchronization calibration is available here. The user confirmed the same prior recording/mounting procedure, but no measured inter-phone clock offset was supplied.',
              f'Timestamp documentation: [Sensor Logger]({TIME_SOURCE}).', '',
              'The existing 250-sample/50-sample-stride preprocessing and fixed motion/axis gate were used only to inspect candidate input windows. A passing window is not a correct classification. No weights were loaded or trained, and no activity accuracy was calculated.',
              f"Across the 8 available pairs, {report['total_windows_passing_gate']}/{report['total_candidate_windows']} candidate windows pass the existing gate. {report['total_windows_failing_motion_requirement']} fail the motion requirement and {report['total_windows_failing_axis_requirement']} fail the axis requirement (these failure counts overlap). The plots show still periods before/after the main movement, and some additional motion before the main sequence; rejected windows are not evidence of missing sensor samples.",
              'All Annotation.csv files are empty. Filenames supply trial-level labels but do not identify standing, transitions, turns, or the exact stair interval. Those portions must not automatically be scored as pure walking/stairs.', '',
              '## Recommended next step', '',
              'Recover walking trial 1 phone 2, retain the device-ID correction for stair ascent trial 2, and establish how the phone clocks were synchronized. Then evaluate the frozen 100 Hz classifier on predefined activity intervals, reporting both classifications and gate coverage. Joint-angle accuracy still requires independent reference angles.', '',
              '[Motion overview](motion_overview.png) · [Pair table](trial_pairs.csv) · [Sensor quality](sensor_quality.csv) · [Full audit](audit.json)', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    # Confirm read-only treatment of the supplied dataset.
    for record in records:
        assert hashlib.sha256((args.source / record['file']).read_bytes()).hexdigest() == record['sha256']
    print(json.dumps({k: v for k, v in report.items() if k not in ['records', 'pairs']}, indent=2))
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()

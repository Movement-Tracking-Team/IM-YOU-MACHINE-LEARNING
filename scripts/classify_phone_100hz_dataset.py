"""Frozen classifier evaluation of audited paired Sensor Logger recordings."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomat.two_sensor_100hz import OUT, HZ, LENGTH, PREPROCESS, SITES, canonicalize, new_model, predict
from biomat.classifier import CLASSES
from audit_phone_100hz_dataset import interpolate, ACTIVITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT / 'reports/phone_100hz_audit/audit.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/phone_100hz_classifier')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    source = Path(audit['source_directory'])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / 'final/classifier/model.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inputs = {r['file']: r['sha256'] for r in audit['records']}
    actual_files = {p.name for p in source.glob('*.zip')}
    if actual_files != set(inputs):
        raise ValueError('Dataset inventory changed; rerun the audit first')
    for name, expected in inputs.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Dataset changed after audit: ' + name)
    protocol = {
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': digest,
        'sample_rate_hz': 100, 'window_length': 250, 'stride_samples': 50,
        'feature_order': 'shank: accel XYZ then gyro XYZ; thigh: accel XYZ then gyro XYZ',
        'source_acceleration': 'AccelerometerUncalibrated.csv; g; gravity included',
        'source_gyro': 'GyroscopeUncalibrated.csv; rad/s; no unit conversion applied',
        'alignment': 'Independent sensor timestamps interpolated onto shared 100Hz overlap grid; integer epoch ns differenced before float conversion. No clock shift fitted.',
        'filter': '4th-order 20Hz Butterworth forward/backward on full aligned trial',
        'normalization': PREPROCESS,
        'gate': 'Each phone >=75% samples gyro norm >0.25rad/s; transverse principal-axis fraction >=0.60',
        'scaler': 'Loaded fixed Camargo training mean and scale from checkpoint',
        'evaluation': 'Agreement with user-supplied trial-level activity among accepted windows; not verified frame-level accuracy. Retain all predictions and rejected-window mask.',
        'phone_training_or_tuning': False, 'clock_synchronization_verified': False,
        'source_sha256': inputs,
    }
    # Persist this fixed protocol before inference; no selection based on results.
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key, expected in {'task': 'classifier', 'sample_rate': HZ, 'window_length': LENGTH,
                          'sites': SITES, 'classes': CLASSES, 'preprocess': PREPROCESS}.items():
        if saved[key] != expected:
            raise ValueError('Incorrect checkpoint metadata: ' + key)
    model = new_model('classifier')
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    mean, scale = saved['mean'].numpy(), saved['scale'].numpy()
    rows, trials, plot_data = [], [], {}
    for pair in audit['pairs']:
        if pair['missing_phones']:
            trials.append({'activity': pair['activity'], 'trial': pair['trial'], 'status': 'missing_phone_2'})
            continue
        signals = {}
        for phone in (1, 2):
            with zipfile.ZipFile(source / pair[f'phone_{phone}_file']) as archive:
                signals[phone] = {sensor: pd.read_csv(io.BytesIO(archive.read(sensor + '.csv')))
                                  for sensor in ['AccelerometerUncalibrated', 'GyroscopeUncalibrated']}
        origin = max(int(df.time.iloc[0]) for values in signals.values() for df in values.values())
        stop = min(int(df.time.iloc[-1]) for values in signals.values() for df in values.values())
        timeline = np.arange(int(np.floor((stop - origin) / 1e9 * HZ)) + 1) / HZ
        x = np.concatenate([np.column_stack([interpolate(signals[p]['AccelerometerUncalibrated'], timeline, origin),
                                              interpolate(signals[p]['GyroscopeUncalibrated'], timeline, origin)])
                            for p in [2, 1]], axis=1)
        x = sosfiltfilt(butter(4, 20, fs=HZ, output='sos'), x, axis=0)
        starts = np.arange(0, len(x) - LENGTH + 1, 50)
        windows = np.stack([x[start:start + LENGTH] for start in starts])
        canonical, quality = canonicalize(windows)
        usable = np.ones(len(windows), dtype=bool)
        for site in (0, 1):
            fraction = (np.linalg.norm(windows[:, :, site*6+3:site*6+6], axis=2) > .25).mean(1)
            usable &= (fraction >= .75) & (quality[site]['principal_axis_fraction'] >= .6)
        audit_windows = pd.read_csv(args.audit.parent / f"{pair['activity']}_trial_{pair['trial']}_window_quality.csv")
        np.testing.assert_array_equal(usable, audit_windows['passes_existing_gate'].to_numpy())
        np.testing.assert_allclose(timeline[starts], audit_windows['start_s_from_overlap'], rtol=0, atol=1e-10)
        probabilities = predict(model, canonical, mean, scale, 'cpu', 'classifier')
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite classifier outputs')
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        labels = probabilities.argmax(1)
        expected_label = CLASSES.index(pair['activity'])
        accepted_counts = np.bincount(labels[usable], minlength=5)
        winners = np.flatnonzero(accepted_counts == accepted_counts.max()) if usable.any() else []
        dominant = CLASSES[winners[0]] if len(winners) == 1 else ('tie' if len(winners) else 'not_evaluated')
        trial = {'activity': pair['activity'], 'trial': pair['trial'], 'status': 'evaluated',
                 'candidate_windows': len(windows), 'accepted_windows': int(usable.sum()),
                 'rejected_windows': int((~usable).sum()),
                 'matching_trial_label_windows': int(np.sum(usable & (labels == expected_label))),
                 'trial_label_agreement': float(np.mean(labels[usable] == expected_label)) if usable.any() else None,
                 'dominant_prediction': dominant, 'dominant_prediction_matches_trial_label': dominant == pair['activity'],
                 'accepted_prediction_counts': {name: int(accepted_counts[i]) for i, name in enumerate(CLASSES)},
                 'overlap_duration_s': (stop - origin) / 1e9, 'epoch_origin_ns': origin,
                 'phone_1_file': pair['phone_1_file'], 'phone_2_file': pair['phone_2_file']}
        trials.append(trial)
        destination = args.output / f"{pair['activity']}_trial_{pair['trial']}"
        destination.mkdir(exist_ok=True)
        # Save inputs for independent inference verification, not for training.
        np.savez_compressed(destination / 'predictions.npz', canonical_input=canonical,
                            probabilities=probabilities, usable=usable, start_s=timeline[starts], epoch_origin_ns=origin)
        local_rows = []
        for index, start in enumerate(starts):
            row = {'activity_label': pair['activity'], 'trial': pair['trial'], 'window': index,
                   'start_s_from_overlap': float(timeline[start]), 'end_s_from_overlap': float(timeline[start] + (LENGTH - 1) / HZ),
                   'usable': bool(usable[index]), 'raw_candidate': CLASSES[labels[index]],
                   'reported_prediction': CLASSES[labels[index]] if usable[index] else 'not_evaluated',
                   'matches_trial_label': bool(labels[index] == expected_label) if usable[index] else None,
                   'maximum_uncalibrated_score': float(probabilities[index].max())}
            row.update({name + '_score': float(probabilities[index, i]) for i, name in enumerate(CLASSES)})
            local_rows.append(row)
        pd.DataFrame(local_rows).to_csv(destination / 'windows.csv', index=False)
        rows.extend(local_rows)
        plot_data[(pair['activity'], pair['trial'])] = (timeline[starts] + (LENGTH - 1) / HZ / 2, labels, usable)
        print(json.dumps(trial), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'all_windows.csv', index=False)
    evaluated = [t for t in trials if t['status'] == 'evaluated']
    accepted = table[table['usable']]
    by_activity = {}
    confusion = np.zeros((3, 5), dtype=int)
    for i, activity in enumerate(ACTIVITIES):
        subset = accepted[accepted['activity_label'] == activity]
        counts = Counter(subset['raw_candidate'])
        confusion[i] = [counts.get(c, 0) for c in CLASSES]
        by_activity[activity] = {'accepted_windows': len(subset), 'matching_windows': counts.get(activity, 0),
                                 'trial_label_agreement': counts.get(activity, 0) / len(subset) if len(subset) else None,
                                 'prediction_counts': dict(counts)}
    confusion_table = pd.DataFrame(confusion, index=ACTIVITIES, columns=CLASSES)
    confusion_table.index.name = 'trial_activity_label'
    confusion_table.to_csv(args.output / 'confusion_counts.csv')
    summary = {'protocol': protocol, 'trials': trials, 'by_activity': by_activity,
               'complete_trials': len(evaluated), 'missing_trials': [t for t in trials if t['status'] != 'evaluated'],
               'candidate_windows': len(table), 'accepted_windows': len(accepted), 'rejected_windows': int((~table['usable']).sum()),
               'matching_trial_label_windows': int((accepted.raw_candidate == accepted.activity_label).sum()),
               'trial_label_agreement': float((accepted.raw_candidate == accepted.activity_label).mean()),
               'trials_with_matching_dominant_prediction': sum(t['dominant_prediction_matches_trial_label'] for t in evaluated),
               'frame_level_accuracy_validated': False,
               'limitations': ['Only trial-level labels; no timed annotations for turns or transitions.',
                               'Epoch alignment assumes comparable device clocks; no measured cross-device clock correction.',
                               'Overlapping windows are correlated; 182 windows are not 182 independent trials.',
                               'Gate coverage must accompany label agreement; rejected windows have no reported activity.',
                               'No fine-tuning, scaler fitting, sign selection, threshold tuning, or temporal offset search on these data.',
                               'This evaluates the classifier, not BioMAT joint-angle accuracy.']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    pd.DataFrame([{k: v for k, v in t.items() if k not in ['accepted_prediction_counts', 'phone_1_file', 'phone_2_file']}
                  for t in trials]).to_csv(args.output / 'trial_results.csv', index=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    colors = ['#2878b5', '#8e63ad', '#c073ad', '#22956c', '#d07821']
    display_names = ['Walk', 'Ramp up', 'Ramp down', 'Stairs up', 'Stairs down']
    for i, activity in enumerate(ACTIVITIES):
        for j, trial in enumerate([1, 2, 3]):
            ax = axes[i, j]
            ax.set_title(f'{activity.replace("_", " ").title()} — trial {trial}')
            if (activity, trial) not in plot_data:
                ax.text(.5, .5, 'Missing phone 2', ha='center', va='center', transform=ax.transAxes)
            else:
                time, labels, usable = plot_data[(activity, trial)]
                ax.axhline(CLASSES.index(activity), color='black', linestyle='--', alpha=.4, label='Trial label')
                ax.scatter(time[~usable], np.full((~usable).sum(), -1), color='#bbbbbb', marker='x', s=16)
                for label in range(5):
                    mask = usable & (labels == label)
                    ax.scatter(time[mask], labels[mask], s=22, color=colors[label])
            ax.set_yticks(range(-1, 5), ['Rejected'] + display_names, fontsize=8)
            ax.set_ylim(-1.5, 4.6)
            ax.set_xlabel('Window center, seconds from overlap')
            ax.grid(alpha=.15)
    fig.suptitle('Frozen 100 Hz classifier on phones — trial labels are not timed ground truth', fontsize=14)
    fig.savefig(args.output / 'prediction_timeline.png', dpi=150)
    plt.close(fig)

    lines = ['# Frozen 100 Hz classifier on collected phone recordings', '',
             f"**Agreement with trial labels: {summary['matching_trial_label_windows']}/{len(accepted)} accepted windows ({summary['trial_label_agreement']:.1%}).**",
             f"{len(accepted)}/{len(table)} candidate windows passed the unchanged motion/axis gate; {summary['rejected_windows']} were rejected.",
             f"The dominant accepted prediction matched the trial label in {summary['trials_with_matching_dominant_prediction']}/{len(evaluated)} complete trials.", '',
             'These figures describe trial-label agreement, not validated frame-by-frame accuracy. All predictions are from the existing classifier; nothing was retrained or tuned using the phone recordings.', '',
             '| Trial label | Trial | Accepted / all windows | Matching / accepted | Agreement | Dominant prediction |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for t in trials:
        if t['status'] != 'evaluated':
            lines.append(f"| {t['activity']} | {t['trial']} | Missing phone 2 | — | — | — |")
        else:
            fraction = f"{t['trial_label_agreement']:.1%}" if t['trial_label_agreement'] is not None else 'N/A'
            lines.append(f"| {t['activity']} | {t['trial']} | {t['accepted_windows']}/{t['candidate_windows']} | {t['matching_trial_label_windows']}/{t['accepted_windows']} | {fraction} | {t['dominant_prediction']} |")
    lines += ['', '## Accepted prediction counts', '',
              '| Trial label | Walking | Ramp up | Ramp down | Stairs up | Stairs down |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for activity in ACTIVITIES:
        lines.append('| ' + activity + ' | ' + ' | '.join(str(int(v)) for v in confusion_table.loc[activity]) + ' |')
    lines += ['', '## Fixed input preparation', '',
              'Phone 1 is right thigh, phone 2 is right shank. The audit resolves the mislabeled stair-ascent trial 2 ZIP using device identity. Walking trial 1 is skipped because phone 2 is missing.',
              'Inputs are raw gravity-inclusive acceleration in g and raw gyro in rad/s. Axes are selected by name x/y/z. Each independent sensor stream is interpolated by integer-differenced epoch timestamps onto its paired 100 Hz overlap; no temporal shift or label-based alignment is fitted. Filtering is 20 Hz forward/backward. Windows contain 250 samples and advance 50. Canonicalization and training-population scaling match canonical100_v1.',
              'The fixed gate requires both sensors to have at least 75% of samples above 0.25 rad/s gyro magnitude and a transverse principal-axis fraction >=0.60. Rejected windows retain raw outputs for auditing but are not reported as predictions or included in the label-agreement denominator.',
              f"Checkpoint SHA-256: `{digest}`.", '', '## Interpretation limits', '']
    lines += ['- ' + item for item in summary['limitations']]
    lines += ['', '[Prediction timeline](prediction_timeline.png) · [Every window](all_windows.csv) · [Trial results](trial_results.csv) · [Full results](summary.json) · [Input audit](../phone_100hz_audit/README.md)', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    for name, expected in inputs.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    print(json.dumps({k: v for k, v in summary.items() if k not in ['protocol', 'trials', 'limitations']}, indent=2))


if __name__ == '__main__':
    main()

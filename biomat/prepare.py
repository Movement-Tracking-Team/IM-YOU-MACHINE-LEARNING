"""Convert original MATLAB ZIPs to BioMAT windows, preserving upstream rules."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from matio import load_from_mat
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Camargo et al. Dataset'
CACHE = DATA / 'biomat_reproduction' / 'windows_v1'
FEATURES = [f'{s}_{f}' for s in ['foot', 'shank', 'thigh']
            for f in ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
TARGETS = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
TRAIN = [f'AB{i:02d}' for i in range(6, 22)]
TEST = ['AB23', 'AB24', 'AB25']


def scalar_text(value):
    while isinstance(value, (np.ndarray, list, tuple)):
        if np.size(value) != 1:
            raise ValueError(f'Expected one label, received {value!r}')
        value = np.asarray(value).reshape(-1)[0]
    return str(value)


def activity_segments(labels, mode):
    """Match upstream all_split, including discarding central level-ground loop."""
    labels = np.asarray([scalar_text(x) for x in labels])
    if mode == 'levelground':
        t1, t2 = [np.flatnonzero(labels == t) for t in ['turn1', 'turn2']]
        if len(t1) == 0 or len(t2) == 0:
            raise ValueError('Missing turn labels required by upstream level-ground splitting')
        if t1[0] > t2[0]:
            t1, t2 = t2, t1
        walking = np.isin(labels, ['stand-walk', 'walk', 'walk-stand'])
        for name, region in [('levelground1', np.arange(len(labels)) <= t1[-1]),
                             ('levelground2', np.arange(len(labels)) >= t2[0])]:
            yield name, np.flatnonzero(walking & region)
    else:
        for direction in ['ascent', 'descent']:
            name = mode + direction
            yield name, np.flatnonzero(np.isin(labels, [name, 'walk-' + name, name + '-walk']))


def downsample(a):
    # Retain original cubic interpolation/endpoints for this reproduction.
    return interp1d(np.linspace(0, 1, len(a)), a, kind='cubic', axis=0)(
        np.linspace(0, 1, len(a) // 2))


def windows(x, y):
    n = len(x)
    if n == 0:
        raise ValueError('Empty resampled segment')
    starts = [0] if n < 256 else range(0, n - 256 + 1, 128)
    for start in starts:
        valid = min(256, n - start)
        yield (np.pad(x[start:start + valid], ((0, 256 - valid), (0, 0))),
               np.pad(y[start:start + valid], ((0, 256 - valid), (0, 0))), start, valid)


def prepare_subject(subject):
    destination = CACHE / (subject + '.npz')
    manifest = CACHE / (subject + '.json')
    if destination.exists() and manifest.exists():
        return json.loads(manifest.read_text())
    xs, ys, metadata, excluded = [], [], [], []
    trials = 0
    with tempfile.TemporaryDirectory(dir=DATA) as tmp, zipfile.ZipFile(DATA / (subject + '.zip')) as z:
        temp = Path(tmp) / 'table.mat'
        members = sorted(n for n in z.namelist() if '/imu/' in n and n.endswith('.mat')
                         and n.split('/')[-3] in ['levelground', 'ramp', 'stair'])
        for name in members:
            tables = {}
            for sensor in ['imu', 'ik', 'conditions']:
                temp.write_bytes(z.read(name.replace('/imu/', '/' + sensor + '/')))
                tables[sensor] = load_from_mat(temp)['labels' if sensor == 'conditions' else 'data']
            t = tables['imu']['Header'].to_numpy()
            if not all(len(table) == len(t) and np.allclose(table['Header'], t, rtol=0, atol=1e-7)
                       for table in tables.values()):
                raise ValueError(f'Unaligned trial: {subject}/{name}')
            x, y = tables['imu'][FEATURES].to_numpy(), tables['ik'][TARGETS].to_numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Nonfinite trial: {subject}/{name}')
            trials += 1
            for activity, idx in activity_segments(tables['conditions']['Label'], name.split('/')[-3]):
                if len(idx) < 4:
                    excluded.append({'trial': name, 'activity': activity, 'reason': 'fewer_than_four_rows'})
                    continue
                xx, yy = x[idx], y[idx]
                if yy[:, 2].min() < -100 or yy[:, 0].max() > 100:
                    excluded.append({'trial': name, 'activity': activity, 'reason': 'upstream_angle_rule'})
                    continue
                xx, yy = downsample(xx), downsample(yy)
                for wx, wy, start, valid in windows(xx, yy):
                    xs.append(wx.astype(np.float32))
                    ys.append(wy.astype(np.float32))
                    metadata.append({'subject': subject, 'trial': name, 'activity': activity,
                                     'start': start, 'valid': valid})
    np.savez(destination, x=np.stack(xs), y=np.stack(ys),
             valid=np.array([m['valid'] for m in metadata], dtype=np.int32),
             activity=np.array([m['activity'] for m in metadata]))
    result = {'subject': subject, 'raw_trials': trials, 'windows': len(xs),
              'activity_counts': dict(Counter(m['activity'] for m in metadata)),
              'padded_windows': sum(m['valid'] < 256 for m in metadata),
              'excluded_segments': excluded, 'metadata': metadata,
              'features': FEATURES, 'targets': TARGETS, 'preprocessing_version': 1}
    manifest.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=TRAIN + TEST)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    summaries = []
    for subject in args.subjects:
        s = prepare_subject(subject)
        summaries.append({k: v for k, v in s.items() if k != 'metadata'})
        print(f"{subject}: {s['windows']} windows, {len(s['excluded_segments'])} excluded segments", flush=True)
    (CACHE / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('TOTAL_WINDOWS', sum(s['windows'] for s in summaries), flush=True)


if __name__ == '__main__':
    main()

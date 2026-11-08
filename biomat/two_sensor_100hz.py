"""Camargo-only 100 Hz two-IMU experiment; frozen counterpart of canonical60_v1.

No existing three-IMU checkpoints or phone labels are used for fitting.
"""
import argparse
import gc
import json
import random
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import DataLoader, TensorDataset

from .classifier import MovementCNN, LABEL_MAP, CLASSES, fit_scaler, score
from .model import BioMAT
from .prepare import ROOT, DATA, CACHE, TRAIN, TEST, TARGETS
from .train import write_json

OUT = DATA / 'two_sensor' / 'canonical100_v1'
HZ, LENGTH = 100, 250
SITES = ['shank', 'thigh']
PREPROCESS = 'canonical100_v1: 20Hz lowpass; 250 samples; median acceleration norm; mean-acceleration + transverse-gyro PCA basis; peak-projection sign'


def replace_checkpoint(source, destination):
    """Allow transient Windows sync/indexer locks to clear before replacement."""
    for attempt in range(12):
        try:
            source.replace(destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                raise
            print('Checkpoint temporarily locked; retrying:', error, flush=True)
            time.sleep(min(0.2 * (attempt + 1), 2.0))


def canonicalize(x):
    """Same deterministic, label-free rotation/acceleration-scale normalization per window.

This frame is signal-derived, NOT an anatomical calibration. Its sign may change
between windows. All predictions and validation use this same representation.
"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (LENGTH, 12) or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 250 x 12 signals')
    out = np.empty_like(x)
    quality = []
    for site in range(2):
        a, w = x[:, :, site*6:site*6+3], x[:, :, site*6+3:site*6+6]
        scale = np.median(np.linalg.norm(a, axis=2), axis=1)
        if np.any(scale < 1e-8):
            raise ValueError('Cannot normalize zero acceleration')
        a = a / scale[:, None, None]
        vertical = a.mean(1)
        strength = np.linalg.norm(vertical, axis=1)
        if np.any(strength < 1e-6):
            raise ValueError('Mean acceleration does not establish a stable reference')
        vertical /= strength[:, None]
        transverse = w - np.einsum('nti,ni->nt', w, vertical)[:, :, None] * vertical[:, None, :]
        cov = np.einsum('nti,ntj->nij', transverse, transverse) / LENGTH
        eigenvalues, vectors = np.linalg.eigh(cov)
        lateral = vectors[:, :, -1]
        lateral -= np.sum(lateral * vertical, axis=1)[:, None] * vertical
        lateral_norm = np.linalg.norm(lateral, axis=1)
        if np.any(lateral_norm < 1e-8):
            raise ValueError('No transverse rotation reference')
        lateral /= lateral_norm[:, None]
        projection = np.einsum('nti,ni->nt', transverse, lateral)
        peak = projection[np.arange(len(x)), np.abs(projection).argmax(1)]
        lateral *= np.where(peak < 0, -1., 1.)[:, None]
        forward = np.cross(vertical, lateral)
        basis = np.stack([lateral, forward, vertical], axis=2)
        out[:, :, site*6:site*6+3] = np.einsum('nti,nij->ntj', a, basis)
        out[:, :, site*6+3:site*6+6] = np.einsum('nti,nij->ntj', w, basis)
        quality.append({'median_acceleration_norm': scale,
                        'transverse_gyro_rms': np.sqrt(np.maximum(eigenvalues[:, -1], 0)),
                        'principal_axis_fraction': eigenvalues[:, -1] / np.maximum(eigenvalues.sum(1), 1e-12)})
    return out.astype(np.float32), quality


def prepare(subjects):
    folder = OUT / 'prepared'
    folder.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        dest = folder / (subject + '.npz')
        if dest.exists():
            continue
        with np.load(CACHE / (subject + '.npz'), allow_pickle=False) as a:
            keep = a['valid'] == 256
            # Preserve first 2.50 seconds of each original 2.56 second full window.
            x = a['x'][keep, :250, 6:18]
            y = a['y'][keep, :250]
            activity = a['activity'][keep]
        filtered = sosfiltfilt(butter(4, 20, fs=100, output='sos'), x, axis=1)
        # Retain the cached 100 Hz grid: no 100-to-60 Hz resampling.
        canonical, _ = canonicalize(filtered)
        labels = np.array([LABEL_MAP[str(v)] for v in activity], dtype=np.int64)
        np.savez(dest, x=canonical, y=y.astype(np.float32), labels=labels, activity=activity)
        print('Prepared', subject, canonical.shape, flush=True)


def load(subjects):
    rows = []
    for subject in subjects:
        with np.load(OUT / 'prepared' / (subject + '.npz'), allow_pickle=False) as a:
            rows.append((a['x'], a['y'], a['labels'], a['activity'], np.full(len(a['x']), subject)))
    return tuple(np.concatenate([row[i] for row in rows]) for i in range(5))


def reg_score(y, p, subjects, activities):
    error = (p.astype(np.float64) - y) ** 2
    by_subject = {str(s): np.sqrt(error[subjects == s].mean((0, 1))).tolist() for s in np.unique(subjects)}
    return {'targets': TARGETS, 'rmse_degrees': np.sqrt(error.mean((0, 1))).tolist(),
            'mae_degrees': np.abs(p.astype(np.float64)-y).mean((0, 1)).tolist(),
            'mean_rmse_degrees': float(np.sqrt(error.mean((0, 1))).mean()),
            'subject_mean_rmse_degrees': float(np.mean(list(by_subject.values()))),
            'by_subject_rmse_degrees': by_subject,
            'by_activity_rmse_degrees': {str(a): np.sqrt(error[activities == a].mean((0, 1))).tolist()
                                         for a in np.unique(activities)}, 'windows': len(y)}


@torch.inference_mode()
def predict(model, x, mean, scale, device, task):
    model.eval()
    out = []
    for start in range(0, len(x), 64):
        batch = torch.from_numpy((x[start:start+64] - mean) / scale).to(device)
        value = model(batch)
        if task == 'classifier':
            value = value.softmax(-1)
        out.append(value.cpu().numpy())
    return np.concatenate(out)


def new_model(task):
    return MovementCNN(12, 5) if task == 'classifier' else BioMAT(seq_len=LENGTH, features=12)


def fit(task, training, folder, validation=None, epochs=30):
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / 'model.pt'
    if (folder / 'complete.json').exists():
        saved = torch.load(model_path, weights_only=True, map_location='cpu')
        model = new_model(task).cuda()
        model.load_state_dict(saved['state_dict'])
        return model, saved['mean'].numpy(), saved['scale'].numpy(), saved['epoch']
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, angles, labels, _, _ = training
    target = labels if task == 'classifier' else angles
    mean, scale = fit_scaler(x)
    model = new_model(task).cuda()
    generator = torch.Generator().manual_seed(42)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001) if task == 'classifier'
                 else torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001))
    loader = DataLoader(TensorDataset(torch.from_numpy((x-mean)/scale), torch.from_numpy(target)),
                        batch_size=128 if task == 'classifier' else 50, shuffle=True, generator=generator)
    weights = torch.tensor(len(labels)/(5*np.bincount(labels, minlength=5)), dtype=torch.float32, device='cuda')
    start, best, best_epoch, stale = 0, float('inf'), 0, 0
    latest = folder / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['loader_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, best, best_epoch, stale = state['epoch'], state['best'], state['best_epoch'], state['stale']
    print(folder.parent.name, task, 'starting epoch', start+1, 'of', epochs, flush=True)
    for epoch in range(start+1, epochs+1):
        if task == 'regressor' and validation is not None and stale >= 8:
            break
        started = time.monotonic()
        model.train()
        total = 0.
        for bx, by in loader:
            bx, by = bx.cuda(), by.cuda()
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = (torch.nn.functional.cross_entropy(pred, by, weight=weights) if task == 'classifier'
                    else torch.nn.functional.mse_loss(pred, by))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        row = {'epoch': epoch, 'loss': total/len(x), 'seconds': time.monotonic()-started}
        improved = False
        if validation is not None:
            vx, vy, vl, va, vs = validation
            if task == 'regressor' or epoch == 1 or epoch % 5 == 0:
                vp = predict(model, vx, mean, scale, 'cuda', task)
                if task == 'regressor':
                    val = reg_score(vy, vp, vs, va)['subject_mean_rmse_degrees']
                    row['validation_subject_mean_rmse_degrees'] = val
                    improved = val < best
                    if improved:
                        best, best_epoch, stale = val, epoch, 0
                    else:
                        stale += 1
                else:
                    row['validation_accuracy'] = score(vl, vp, vs)['accuracy']
        if task == 'classifier' or validation is None or improved:
            torch.save({'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                        'epoch': epoch, 'task': task, 'sites': SITES, 'sample_rate': HZ, 'window_length': LENGTH,
                        'preprocess': PREPROCESS, 'classes': CLASSES, 'targets': TARGETS}, model_path)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best': best if np.isfinite(best) else 1e30, 'best_epoch': best_epoch, 'stale': stale,
                    'loader_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, folder / 'latest.tmp')
        replace_checkpoint(folder / 'latest.tmp', latest)
        with (folder / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        write_json(folder / 'status.json', {'state': 'training', **row})
        print(folder.parent.name, task, json.dumps(row), flush=True)
    saved = torch.load(model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(saved['state_dict'])
    write_json(folder / 'complete.json', {'selected_epoch': saved['epoch'], 'last_epoch': epoch if epochs > start else start})
    write_json(folder / 'status.json', {'state': 'complete', 'selected_epoch': saved['epoch']})
    return model, mean, scale, saved['epoch']


def run(tasks=('classifier', 'regressor')):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    OUT.mkdir(parents=True, exist_ok=True)
    folds = json.loads((ROOT / 'reports/biomat_subject_folds.json').read_text())['folds']
    protocol = {
        'version': 1, 'train': TRAIN, 'test': TEST, 'folds': folds, 'sites': SITES,
        'sample_rate_hz': HZ, 'window_length': LENGTH, 'features': 12,
        'preprocessing': PREPROCESS, 'classifier_epochs': 30, 'regressor_max_epochs': 50,
        'regressor_selection': 'CV best subject-mean joint RMSE, patience 8; final median best epoch rounded up',
        'seed': 42,
        'regressor': 'BioMAT d_model256 heads16 d_ff128 layers4 dropout0.5, seq_len250 features12; all 3 original angle heads',
        'scope': 'Camargo only; full curated activity windows; no idle class. No phone data or labels used in fitting, scaling, or selection.',
        'data': 'Existing windows_v1 nominal 100Hz cache, inherited original cubic 200-to-100Hz interpolation and target exclusions; keep first 250 of 256 samples; no conversion to 60Hz.',
        'filter': 'Fourth-order 20Hz Butterworth forward/backward, on each IMU window only; angle targets retained unchanged from 100Hz cache.',
        'acceleration': 'Per-window median vector-norm scale; source acceleration physical-unit discrepancy remains unresolved.',
        'gyro': 'Camargo rad/s per scripts.zip/lib/yLabelSelector.m.',
        'orientation': 'Same signal-derived frame as canonical60_v1, using 250 samples; not anatomical calibration.',
        'initialization': 'From scratch; no 60Hz weights reused.',
        'test_history': 'AB23/24/25 evaluated in prior experiments; held out from this fit, not a new untouched study cohort.',
    }
    protocol_file = OUT / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Protocol differs from existing 100Hz run')
    write_json(OUT / 'protocol.json', protocol)
    print('100 Hz Camargo | shank + thigh | 250 samples x 12 channels | CUDA:',
          torch.cuda.get_device_name(), flush=True)
    prepare(TRAIN + TEST)
    # Finish and report the classifier first, then the angle regressor.
    # Each task completes its subject CV before its final held-out evaluation.
    for task in tasks:
        result_file = OUT / ('results_' + task + '.json')
        if result_file.exists():
            print(task, 'already complete', result_file, flush=True)
            continue
        records = []
        for fold in folds:
            folder = OUT / f"fold_{fold['fold']}" / task
            metrics_path = folder / 'metrics.json'
            if metrics_path.exists():
                records.append(json.loads(metrics_path.read_text()))
                continue
            assert not set(fold['train_subjects']) & set(fold['validation_subjects'])
            assert not set(fold['train_subjects'] + fold['validation_subjects']) & set(TEST)
            training = load(fold['train_subjects'])
            validation = load(fold['validation_subjects'])
            vx, vy, vl, va, vs = validation
            model, mean, scale, epoch = fit(task, training, folder, validation,
                                           30 if task == 'classifier' else 50)
            predictions = predict(model, vx, mean, scale, 'cuda', task)
            metrics = score(vl, predictions, vs) if task == 'classifier' else reg_score(vy, predictions, vs, va)
            record = {'fold': fold['fold'], 'selected_epoch': epoch, 'metrics': metrics}
            np.savez(folder / 'predictions.npz', predictions=predictions, labels=vl,
                     angles=vy, subjects=vs, activity=va)
            write_json(metrics_path, record)
            records.append(record)
            del model, training, validation, predictions
            gc.collect()
            torch.cuda.empty_cache()
        final_epochs = 30 if task == 'classifier' else int(np.ceil(np.median([r['selected_epoch'] for r in records])))
        write_json(OUT / ('selection_' + task + '.json'), {
            'final_epochs': final_epochs, 'cv': records,
            'rule': 'Fixed 30 epochs' if task == 'classifier' else protocol['regressor_selection']})
        training = load(TRAIN)
        folder = OUT / 'final' / task
        model, mean, scale, epoch = fit(task, training, folder, epochs=final_epochs)
        tx, ty, tl, ta, ts = load(TEST)
        predictions = predict(model, tx, mean, scale, 'cuda', task)
        metrics = score(tl, predictions, ts) if task == 'classifier' else reg_score(ty, predictions, ts, ta)
        np.savez(folder / 'predictions.npz', predictions=predictions, labels=tl,
                 angles=ty, subjects=ts, activity=ta)
        write_json(result_file, {'protocol': protocol, 'task': task, 'cv': records,
                                'final_epochs': epoch, 'test': metrics})
        print('COMPLETED', task, json.dumps(metrics), flush=True)
        del model, training, predictions, tx, ty, tl, ta, ts
        gc.collect()
        torch.cuda.empty_cache()
    result_files = [OUT / ('results_' + task + '.json') for task in ['classifier', 'regressor']]
    if all(path.exists() for path in result_files):
        c, r = [json.loads(path.read_text()) for path in result_files]
        combined_cv = []
        for classifier, regressor in zip(c['cv'], r['cv']):
            assert classifier['fold'] == regressor['fold']
            combined_cv.append({'fold': classifier['fold'],
                                'classifier': {**classifier['metrics'], 'selected_epoch': classifier['selected_epoch']},
                                'regressor': {**regressor['metrics'], 'selected_epoch': regressor['selected_epoch']}})
        write_json(OUT / 'results.json', {'protocol': protocol, 'cv': combined_cv,
                                        'final_regressor_epochs': r['final_epochs'],
                                        'test': {'classifier': c['test'], 'regressor': r['test']}})
        print('COMPLETE: both 100 Hz models and held-out Camargo results saved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['classifier', 'regressor', 'both'], default='both')
    args = parser.parse_args()
    run(('classifier', 'regressor') if args.task == 'both' else (args.task,))


"""Fixed-protocol five-fold CV and a final held-out BioMAT reproduction run."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import BioMAT
from .prepare import ROOT, DATA, CACHE, TRAIN, TEST, FEATURES, TARGETS

RUNS = DATA / 'biomat_reproduction' / 'runs'


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def load_subjects(subjects):
    arrays = []
    for subject in subjects:
        with np.load(CACHE / (subject + '.npz'), allow_pickle=False) as a:
            arrays.append((a['x'], a['y'], a['valid'], a['activity'],
                           np.full(len(a['x']), subject)))
    return tuple(np.concatenate([a[i] for a in arrays]) for i in range(5))


def metrics(pred, target, valid, activities, subjects):
    error = (pred.astype(np.float64) - target) ** 2
    mask = np.arange(256)[None, :] < valid[:, None]
    pooled = np.sqrt(error.mean(axis=(0, 1)))
    masked = np.sqrt(error[mask].mean(axis=0))
    rows = []
    activity5 = np.array(['levelground' if s.startswith('levelground') else s for s in activities])
    for subject in sorted(set(subjects)):
        for activity in sorted(set(activity5)):
            idx = (subjects == subject) & (activity5 == activity)
            if not idx.any():
                continue
            rmse = np.sqrt(error[idx].mean(axis=(0, 1)))
            rows.append({'subject': str(subject), 'activity': str(activity),
                         'windows': int(idx.sum()), 'rmse_degrees': rmse.tolist(),
                         'mean_rmse_degrees': float(rmse.mean())})
    subject_means = {s: float(np.mean([r['mean_rmse_degrees'] for r in rows if r['subject'] == s]))
                     for s in sorted(set(subjects))}
    return {'targets': TARGETS, 'pooled_rmse_degrees': pooled.tolist(),
            'pooled_mean_rmse_degrees': float(pooled.mean()),
            'valid_frames_rmse_degrees': masked.tolist(),
            'subject_activity_macro_rmse_degrees': float(np.mean(list(subject_means.values()))),
            'subject_mean_rmse_degrees': subject_means,
            'subject_std_degrees': float(np.std(list(subject_means.values()))),
            'by_subject_activity': rows}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for x, _ in loader:
        predictions.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(predictions)


def run_one(name, train_subjects, eval_subjects, args):
    run = RUNS / args.experiment / name
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'metrics.json').exists():
        print(f'{name}: already complete', flush=True)
        return json.loads((run / 'metrics.json').read_text())
    assert not set(train_subjects) & set(eval_subjects)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    x, y, _, _, _ = load_subjects(train_subjects)
    ex, ey, valid, activities, subjects = load_subjects(eval_subjects)
    # Match StandardScaler over all training windows, including original padding.
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1
    if args.scaling == 'standard':
        x = (x - mean) / scale
        ex = (ex - mean) / scale
    np.savez(run / 'scaler.npz', mean=mean, scale=scale, features=np.array(FEATURES))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=True)
    eval_loader = DataLoader(TensorDataset(torch.from_numpy(ex), torch.from_numpy(ey)),
                             batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = BioMAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    protocol = {'name': name, 'train_subjects': train_subjects, 'evaluation_subjects': eval_subjects,
                'epochs': args.epochs, 'batch_size': args.batch_size, 'scaling': args.scaling,
                'seed': args.seed, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(), 'parameters': sum(p.numel() for p in model.parameters()),
                'train_windows': len(x), 'evaluation_windows': len(ex),
                'model': {'d_model': 256, 'heads': 16, 'd_ff': 128, 'layers': 4, 'dropout': 0.5},
                'optimizer': {'name': 'Adam', 'lr': 0.001, 'weight_decay': 0.001},
                'precision': 'float32; no AMP; no TF32',
                'evaluation_policy': 'Final epoch; held-out test evaluated only after training',
                'loss': 'MSE including padding, as upstream', 'preprocessing_version': 1,
                'upstream_commit': '0ca003db504514da1a45d2a2d9339934f6b68202'}
    if (run / 'protocol.json').exists() and json.loads((run / 'protocol.json').read_text()) != protocol:
        raise ValueError('Existing run protocol differs; choose a new experiment name')
    write_json(run / 'protocol.json', protocol)
    checkpoint = run / 'latest.pt'
    start_epoch = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['loader_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch = state['epoch']
        del state
    print(f'{name}: {len(x)} train / {len(ex)} eval windows; starting epoch {start_epoch + 1}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.train()
        loss_sum, count = 0.0, 0
        for batch, (bx, by) in enumerate(loader):
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(bx), by)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {name} epoch {epoch + 1}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(bx)
            count += len(bx)
            if args.smoke_batches and batch + 1 >= args.smoke_batches:
                break
        record = {'epoch': epoch + 1, 'train_mse': loss_sum / count,
                  'epoch_seconds': time.monotonic() - started,
                  'gpu_peak_GB': torch.cuda.max_memory_allocated() / 1e9}
        # CV can be monitored. The final holdout is never evaluated here.
        if name.startswith('fold_') and not args.smoke_batches:
            pred = predict(model, eval_loader, device)
            record['validation_pooled_rmse'] = float(np.sqrt(((pred - ey) ** 2).mean(axis=(0, 1))).mean())
        with (run / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        write_json(run / 'status.json', {**record, 'state': 'training', 'total_epochs': args.epochs})
        print(name, json.dumps(record), flush=True)
        torch.save({'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'loader_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, run / 'latest.tmp')
        os.replace(run / 'latest.tmp', checkpoint)
    if args.smoke_batches:
        result = {'smoke_only': True, 'batches': args.smoke_batches, **record}
    else:
        pred = predict(model, eval_loader, device)
        result = metrics(pred, ey, valid, activities, subjects)
        np.savez(run / 'predictions.npz', predictions=pred, targets=ey, valid=valid,
                 activities=activities, subjects=subjects)
    torch.save({'model': model.state_dict(), 'protocol': protocol, 'mean': torch.from_numpy(mean),
                'scale': torch.from_numpy(scale)}, run / 'model.pt')
    write_json(run / 'metrics.json', result)
    write_json(run / 'status.json', {'state': 'complete', 'epochs': args.epochs, 'metrics': result})
    del model, optimizer, loader, eval_loader, x, y, ex, ey
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'cv', 'final', 'all'], default='all')
    parser.add_argument('--experiment', default='paper_standard_v1')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scaling', choices=['standard', 'raw'], default='standard')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--smoke-batches', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this training run')
    if args.mode == 'smoke':
        args.epochs, args.smoke_batches = 1, 3
        args.experiment = 'smoke_' + args.experiment
        run_one('smoke', ['AB06'], ['AB07'], args)
        return
    folds = json.loads((ROOT / 'reports/biomat_subject_folds.json').read_text())['folds']
    results = {}
    if args.mode in ['cv', 'all']:
        for fold in folds:
            if args.fold and fold['fold'] != args.fold:
                continue
            name = 'fold_' + str(fold['fold'])
            results[name] = run_one(name, fold['train_subjects'], fold['validation_subjects'], args)
        write_json(RUNS / args.experiment / 'cv_metrics.json', results)
    if args.mode in ['final', 'all']:
        results['final'] = run_one('final', TRAIN, TEST, args)
    write_json(RUNS / args.experiment / 'summary.json', results)


if __name__ == '__main__':
    main()

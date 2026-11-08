"""Subject-disjoint UCI HAR benchmark of the adapted MovementCNN architecture.

Trained from scratch on phone data: this is not Camargo-to-phone weight transfer.
Uses supplied time sequences, not the dataset's engineered 561-feature vectors.
"""
import hashlib
import io
import json
import random
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .classifier import MovementCNN, fit_scaler, cnn_probabilities
from .prepare import ROOT, DATA
from .train import write_json

SOURCE = 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'
DOWNLOAD = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
SHA256 = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'
BASE = DATA / 'phone_benchmark' / 'uci_har'
OUT = BASE / 'cnn_v1'
CHANNELS = [f'{kind}_{axis}' for kind in ['total_acc', 'body_gyro'] for axis in 'xyz']
CLASSES = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'lying']
EPOCHS = 30


def load_dataset():
    BASE.mkdir(parents=True, exist_ok=True)
    archive = BASE / 'uci_har.zip'
    if not archive.exists():
        temporary = BASE / 'download.partial'
        urllib.request.urlretrieve(DOWNLOAD, temporary)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Dataset archive checksum changed; inspect before proceeding')
    data = {}
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip'))) as inner:
            (BASE / 'original_README.txt').write_bytes(inner.read('UCI HAR Dataset/README.txt'))
            for split in ['train', 'test']:
                prefix = f'UCI HAR Dataset/{split}/'
                def read(name, dtype):
                    return np.loadtxt(io.BytesIO(inner.read(prefix + name)), dtype=dtype)
                x = np.stack([read(f'Inertial Signals/{c}_{split}.txt', np.float32)
                              for c in CHANNELS], axis=-1)
                y = read(f'y_{split}.txt', np.int64) - 1
                subjects = read(f'subject_{split}.txt', np.int64)
                if x.shape != (len(y), 128, 6) or len(subjects) != len(y):
                    raise ValueError('Unexpected dataset shape')
                if not np.isfinite(x).all() or set(y) != set(range(6)):
                    raise ValueError('Invalid signals or labels')
                data[split] = x, y, subjects
    if set(data['train'][2]) & set(data['test'][2]):
        raise ValueError('Official split contains overlapping participants')
    hashes = [{hashlib.sha256(row.tobytes()).digest() for row in data[s][0]} for s in ['train', 'test']]
    if hashes[0] & hashes[1]:
        raise ValueError('Exact signal duplicates cross the official split')
    return data


def make_folds(subjects):
    people = np.unique(subjects)
    return [{'fold': i, 'train_subjects': people[a].tolist(), 'validation_subjects': people[b].tolist()}
            for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=42).split(people), 1)]


def score(y, probabilities, subjects):
    pred = probabilities.argmax(1)
    per_subject = {str(s): {'accuracy': float(accuracy_score(y[subjects == s], pred[subjects == s])),
                          'macro_f1': float(f1_score(y[subjects == s], pred[subjects == s],
                                                    labels=range(6), average='macro', zero_division=0))}
                   for s in np.unique(subjects)}
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(6), average='macro', zero_division=0)),
            'subject_mean_macro_f1': float(np.mean([v['macro_f1'] for v in per_subject.values()])),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(6)).tolist(),
            'classification_report': classification_report(y, pred, labels=range(6), target_names=CLASSES,
                                                           output_dict=True, zero_division=0),
            'subjects': per_subject, 'windows': len(y)}


def train(x, y, folder, validation=None):
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MovementCNN(input_channels=6, num_classes=6).to(device)
    mean, scale = fit_scaler(x)
    counts = np.bincount(y, minlength=6)
    if np.any(counts == 0):
        raise ValueError('Missing training class')
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (6 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(TensorDataset(torch.from_numpy((x - mean) / scale), torch.from_numpy(y)),
                        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42))
    # An interrupted fold restarts from its seeded initialization; never appends duplicate epochs.
    with (folder / 'epochs.jsonl').open('w') as log:
        for epoch in range(1, EPOCHS + 1):
            started = time.monotonic()
            model.train()
            total = 0.0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(bx), by)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                optimizer.step()
                total += loss.item() * len(bx)
            row = {'epoch': epoch, 'loss': total / len(y), 'seconds': time.monotonic() - started}
            if validation is not None and (epoch == 1 or epoch % 5 == 0):
                vx, vy, vs = validation
                metrics = score(vy, cnn_probabilities(model, vx, mean, scale, device), vs)
                row.update(validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1'])
            log.write(json.dumps(row) + '\n')
            log.flush()
            write_json(folder / 'status.json', {'state': 'training', **row})
            print(folder.name, json.dumps(row), flush=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'mean': torch.from_numpy(mean),
                'scale': torch.from_numpy(scale), 'classes': CLASSES, 'channels': CHANNELS,
                'input_channels': 6, 'num_classes': 6, 'sample_rate': 50, 'window_length': 128,
                'acceleration_units': 'g, includes gravity', 'gyroscope_units': 'rad/s',
                'placement': 'waist', 'epochs': EPOCHS, 'source': SOURCE,
                'preprocessing': 'UCI supplied filtered signal windows; train-only channel standardization',
                'trained_from_scratch': True}, folder / 'cnn.pt')
    return model.to(device), mean, scale, device


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        report(json.loads((OUT / 'results.json').read_text()))
        print('Already complete:', OUT / 'results.json', flush=True)
        return
    data = load_dataset()
    x, y, subjects = data['train']
    folds = make_folds(subjects)
    protocol = {'source': SOURCE, 'download': DOWNLOAD, 'archive_sha256': SHA256,
                'seed': 42, 'epochs': EPOCHS, 'classes': CLASSES, 'channels': CHANNELS,
                'sample_rate': 50, 'window_length': 128, 'overlap': 0.5,
                'train_subjects': np.unique(subjects).tolist(), 'test_subjects': np.unique(data['test'][2]).tolist(),
                'training_windows': len(y), 'test_windows': len(data['test'][1]), 'folds': folds,
                'optimizer': 'AdamW lr=0.001 weight_decay=0.0001, inverse-frequency CE, batch 128',
                'parameters': sum(p.numel() for p in MovementCNN(6, 6).parameters()),
                'selection': 'No tuning; fixed 30 epochs inherited from Camargo classifier. Final test evaluated after fitting.',
                'scope': 'Phone-trained architecture generalization to unseen people on UCI HAR; no Camargo weight transfer or new-device validation',
                'preprocessing': 'Supplied filtered total_acc (g, with gravity), body_gyro (rad/s). Train-only standardization.',
                'audit': 'Finite signals, six classes, no participant overlap or exact signal duplicates across official split',
                'torch': torch.__version__, 'numpy': np.__version__,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    write_json(OUT / 'protocol.json', protocol)
    print('UCI HAR PHONE BENCHMARK: new 6-channel CNN, not existing Camargo weights.', flush=True)
    print(json.dumps(protocol), flush=True)
    records = []
    for fold in folds:
        folder = OUT / f"fold_{fold['fold']}"
        if (folder / 'metrics.json').exists():
            records.append(json.loads((folder / 'metrics.json').read_text()))
            continue
        a = np.isin(subjects, fold['train_subjects'])
        b = np.isin(subjects, fold['validation_subjects'])
        assert not np.any(a & b) and np.all(a | b)
        print(folder.name, 'train windows:', int(a.sum()), 'validation:', int(b.sum()), flush=True)
        model, mean, scale, device = train(x[a], y[a], folder, (x[b], y[b], subjects[b]))
        p = cnn_probabilities(model, x[b], mean, scale, device)
        result = {'fold': fold['fold'], **score(y[b], p, subjects[b])}
        np.savez(folder / 'predictions.npz', probabilities=p, labels=y[b], subjects=subjects[b])
        write_json(folder / 'metrics.json', result)
        write_json(folder / 'status.json', {'state': 'complete'})
        records.append(result)
        del model
    cv = {key: float(np.mean([r[key] for r in records]))
          for key in ['accuracy', 'macro_f1', 'subject_mean_macro_f1']}
    write_json(OUT / 'cv_results.json', {'folds': records, 'mean': cv})
    print('CROSS-VALIDATION COMPLETE', json.dumps(cv), flush=True)
    folder = OUT / 'final'
    model, mean, scale, device = train(x, y, folder)
    tx, ty, ts = data['test']
    probabilities = cnn_probabilities(model, tx, mean, scale, device)
    np.savez(folder / 'predictions.npz', probabilities=probabilities, labels=ty, subjects=ts)
    result = {'cv': records, 'cv_mean': cv, 'test': score(ty, probabilities, ts), 'protocol': protocol}
    write_json(OUT / 'results.json', result)
    write_json(folder / 'status.json', {'state': 'complete'})
    report(result)
    print('COMPLETE: held-out phone accuracy', result['test']['accuracy'],
          'macro F1', result['test']['macro_f1'], flush=True)


def report(result):
    test = result['test']
    lines = ['# UCI HAR phone CNN benchmark', '',
             '**This CNN was trained from scratch on phone data. The existing Camargo weights were not evaluated on phones.**', '',
             f"Official test: **{test['accuracy']:.2%} accuracy**, **{test['macro_f1']:.4f} macro-F1**, {test['windows']} windows from 9 unseen participants.", '',
             f"Five-fold development mean: {result['cv_mean']['accuracy']:.2%} accuracy, {result['cv_mean']['macro_f1']:.4f} macro-F1.", '',
             '| Activity | Precision | Recall | F1 | Windows |', '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        r = test['classification_report'][name]
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ['', '| Person | Accuracy | Macro-F1 |', '| --- | ---: | ---: |']
    for s, r in test['subjects'].items():
        lines.append(f"| {s} | {r['accuracy']:.2%} | {r['macro_f1']:.3f} |")
    lines += ['', '## Protocol and interpretation', '',
              'Same three convolution blocks as MovementCNN, with six inputs and six outputs. '
              'Total acceleration includes gravity (g); gyroscope is rad/s. Input is 128 samples at 50 Hz, '
              'from a waist-mounted Samsung Galaxy S II. The supplied sequences have already been filtered and windowed. '
              'No engineered 561-feature vectors or participant IDs enter the CNN.', '',
              'Official 21-person training / 9-person test partition retained. Five shuffled subject folds within training only, '
              'seed 42. Scaling and class weights fitted within each training fold. Fixed 30 epochs, AdamW lr 0.001, '
              'weight decay 0.0001, batch 128, no tuning after test. The test signals are read for data-integrity checks '
              'but never used for normalization, fitting, or model selection. Exact cross-partition duplicates checked.', '',
              'This tests generalization to unseen people in the same phone benchmark. It does not test the original '
              '18-channel Camargo weights, different phone models, leg-mounted phones, arbitrary orientations, '
              'running, or continuous free-living recordings. The original three-phone CSV adapter remains for the '
              'Camargo classifier and must not be used with this six-channel checkpoint. Overlapping windows '
              'are correlated, and softmax confidence is uncalibrated.', '',
              f'[Dataset and attribution]({SOURCE}): Anguita et al., A Public Domain Dataset for Human Activity Recognition Using Smartphones, ESANN 2013. '
              'The downloaded original README is preserved with the local data.', '',
              'Artifacts: `Camargo et al. Dataset/phone_benchmark/uci_har/cnn_v1/`; final checkpoint: `final/cnn.pt`. '
              'Protocol includes archive SHA256, participant folds, software versions, and input conventions.', '',
              '[Confusion matrix and learning curves](phone_benchmark_results.png)', '']
    (ROOT / 'reports/phone_benchmark_results.md').write_text('\n'.join(lines), encoding='utf-8')
    cm = np.asarray(test['confusion_matrix'])
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
    labels = ['Walk', 'Upstairs', 'Downstairs', 'Sit', 'Stand', 'Lie']
    ax.set_xticks(range(6), labels, rotation=35)
    ax.set_yticks(range(6), labels)
    ax.set(xlabel='Predicted', ylabel='Actual', title=f"Unseen phone users: {test['accuracy']:.1%} accuracy")
    for i in range(6):
        for j in range(6):
            fraction = cm[i, j] / cm[i].sum()
            ax.text(j, i, f'{cm[i,j]}\n{fraction:.0%}', ha='center', va='center',
                    color='white' if fraction > .55 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.7, label='Fraction of actual activity')
    for record in result['cv']:
        rows = [json.loads(s) for s in (OUT / f"fold_{record['fold']}" / 'epochs.jsonl').read_text().splitlines()]
        rows = [r for r in rows if 'validation_accuracy' in r]
        curve.plot([r['epoch'] for r in rows], [100*r['validation_accuracy'] for r in rows],
                   marker='.', label=f"Fold {record['fold']}")
    curve.set(xlabel='Epoch', ylabel='Validation accuracy (%)', title='Validation: participants kept separate', ylim=(0, 100))
    curve.legend()
    curve.grid(alpha=.2)
    fig.savefig(ROOT / 'reports/phone_benchmark_results.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

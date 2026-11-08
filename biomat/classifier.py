"""Five-class activity-window experiment; subject-disjoint model selection."""
import json
from pathlib import Path
import random
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .prepare import ROOT, DATA, CACHE, TRAIN, TEST, FEATURES
from .train import write_json

CLASSES = ['walking', 'ramp_ascent', 'ramp_descent', 'stair_ascent', 'stair_descent']
LABEL_MAP = {'levelground1': 0, 'levelground2': 0, 'rampascent': 1, 'rampdescent': 2,
             'stairascent': 3, 'stairdescent': 4}
OUT = DATA / 'movement_classifier' / 'v1'
EPOCHS = 30


def load_data(subjects):
    xs, ys, people = [], [], []
    for subject in subjects:
        with np.load(CACHE / (subject + '.npz'), allow_pickle=False) as a:
            keep = a['valid'] == 256  # Exclude padding-length cues entirely.
            xs.append(a['x'][keep])
            ys.append(np.array([LABEL_MAP[str(v)] for v in a['activity'][keep]], dtype=np.int64))
            people.append(np.full(int(keep.sum()), subject))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(people)


def features(x):
    return np.concatenate([x.mean(1), x.std(1), np.sqrt((x * x).mean(1)),
                           np.ptp(x, axis=1), np.diff(x, axis=1).std(1)], axis=1)


def fit_scaler(x):
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1
    return mean, scale


class MovementCNN(nn.Module):
    def __init__(self, input_channels=18, num_classes=5):
        super().__init__()
        blocks = []
        for cin, cout, kernel in [(input_channels, 64, 7), (64, 128, 5), (128, 128, 3)]:
            blocks += [nn.Conv1d(cin, cout, kernel, stride=2, padding=kernel // 2),
                       nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(0.2)]
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2)).mean(-1))


def score(y, probabilities, subjects):
    pred = probabilities.argmax(1)
    per_subject = {str(s): {'accuracy': float(accuracy_score(y[subjects == s], pred[subjects == s])),
                           'macro_f1': float(f1_score(y[subjects == s], pred[subjects == s], labels=range(5), average='macro', zero_division=0))}
                   for s in sorted(set(subjects))}
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(5), average='macro', zero_division=0)),
            'subject_mean_macro_f1': float(np.mean([v['macro_f1'] for v in per_subject.values()])),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(5)).tolist(),
            'classification_report': classification_report(y, pred, labels=range(5), target_names=CLASSES,
                                                           output_dict=True, zero_division=0),
            'subjects': per_subject, 'windows': len(y)}


@torch.inference_mode()
def cnn_probabilities(model, x, mean, scale, device):
    model.eval()
    batches = []
    for start in range(0, len(x), 256):
        batch = torch.from_numpy((x[start:start + 256] - mean) / scale).to(device)
        batches.append(model(batch).softmax(-1).cpu().numpy())
    return np.concatenate(batches)


def train_cnn(x, y, folder, validation=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, scale = fit_scaler(x)
    model = MovementCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    counts = np.bincount(y, minlength=5)
    if np.any(counts == 0):
        raise ValueError('A training class is absent')
    weight = torch.tensor(len(y) / (5 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(TensorDataset(torch.from_numpy((x - mean) / scale), torch.from_numpy(y)),
                        batch_size=128, shuffle=True, generator=generator)
    for epoch in range(1, EPOCHS + 1):
        started = time.monotonic()
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classification loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
        row = {'epoch': epoch, 'loss': total / len(x), 'seconds': time.monotonic() - started}
        if validation is not None and (epoch % 5 == 0 or epoch == 1):
            vx, vy, vs = validation
            scores = score(vy, cnn_probabilities(model, vx, mean, scale, device), vs)
            row.update({'validation_accuracy': scores['accuracy'], 'validation_macro_f1': scores['macro_f1']})
        with (folder / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        write_json(folder / 'status.json', {'state': 'training', **row, 'total_epochs': EPOCHS})
        print(folder.name, json.dumps(row), flush=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'mean': torch.from_numpy(mean),
                'scale': torch.from_numpy(scale), 'classes': CLASSES, 'features': FEATURES,
                'sample_rate': 100, 'window_length': 256, 'epochs': EPOCHS}, folder / 'cnn.pt')
    return model.to(device), mean, scale, device


def fit_tree(x, y):
    model = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', class_weight='balanced',
                                random_state=42, n_jobs=4)
    return model.fit(features(x), y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        print('Experiment already complete:', OUT / 'results.json', flush=True)
        return
    folds = json.loads((ROOT / 'reports/biomat_subject_folds.json').read_text())['folds']
    protocol = {'classes': CLASSES, 'features': FEATURES, 'window_length': 256, 'nominal_hz': 100,
                'train_subjects': TRAIN, 'final_test_subjects': TEST, 'folds': folds,
                'selection': 'Largest mean subject-balanced macro-F1 over five folds; fixed epoch 30 for CNN',
                'seed': 42, 'cnn_epochs': EPOCHS, 'cnn_parameters': sum(p.numel() for p in MovementCNN().parameters()),
                'cnn_optimizer': 'AdamW lr=0.001 weight_decay=0.0001; class-weighted CE; batch 128',
                'tree': '300 Extra Trees; balanced class weights; 90 summary features',
                'scope': 'Full activity-segment windows only; transitions grouped into related activity; no idle/running class',
                'inherited_preprocessing': 'BioMAT windows_v1 including target-based exclusion and cubic interpolation',
                'phone_validation': False}
    write_json(OUT / 'protocol.json', protocol)
    print('Movement classifier: CNN versus Extra Trees, five subject folds, then final test.', flush=True)
    print('CNN parameters:', protocol['cnn_parameters'], flush=True)
    cv = []
    for fold in folds:
        folder = OUT / f"fold_{fold['fold']}"
        folder.mkdir(exist_ok=True)
        if (folder / 'metrics.json').exists():
            cv.append(json.loads((folder / 'metrics.json').read_text()))
            continue
        assert not set(fold['train_subjects']) & set(fold['validation_subjects'])
        assert not set(fold['train_subjects'] + fold['validation_subjects']) & set(TEST)
        x, y, _ = load_data(fold['train_subjects'])
        vx, vy, vs = load_data(fold['validation_subjects'])
        print(folder.name, 'training windows', len(x), 'validation windows', len(vx), flush=True)
        tree = fit_tree(x, y)
        tree_result = score(vy, tree.predict_proba(features(vx)), vs)
        print(folder.name, 'Extra Trees', tree_result['accuracy'], tree_result['macro_f1'], flush=True)
        model, mean, scale, device = train_cnn(x, y, folder, (vx, vy, vs))
        probs = cnn_probabilities(model, vx, mean, scale, device)
        record = {'fold': fold['fold'], 'cnn': score(vy, probs, vs), 'extra_trees': tree_result}
        write_json(folder / 'metrics.json', record)
        write_json(folder / 'status.json', {'state': 'complete', 'epochs': EPOCHS})
        cv.append(record)
        del model, tree, x, vx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selection_scores = {name: float(np.mean([f[name]['subject_mean_macro_f1'] for f in cv]))
                        for name in ['cnn', 'extra_trees']}
    selected = max(selection_scores, key=selection_scores.get)
    # Persist decision before reading the final test set.
    write_json(OUT / 'selection.json', {'selected': selected, 'cv_subject_macro_f1': selection_scores})
    print('Selected from cross-validation:', selected, selection_scores, flush=True)
    folder = OUT / 'final'
    folder.mkdir(exist_ok=True)
    x, y, _ = load_data(TRAIN)
    model, mean, scale, device = train_cnn(x, y, folder)
    tree = fit_tree(x, y)
    joblib.dump({'model': tree, 'classes': CLASSES, 'features': FEATURES}, folder / 'extra_trees.joblib')
    tx, ty, ts = load_data(TEST)
    probs = {'cnn': cnn_probabilities(model, tx, mean, scale, device),
             'extra_trees': tree.predict_proba(features(tx))}
    tests = {name: score(ty, p, ts) for name, p in probs.items()}
    for name, p in probs.items():
        np.savez(folder / f'{name}_predictions.npz', probabilities=p, labels=ty, subjects=ts)
    result = {'selected_before_test': selected, 'cv_selection_scores': selection_scores,
              'cv': cv, 'final_test': tests}
    write_json(OUT / 'results.json', result)
    write_json(folder / 'status.json', {'state': 'complete', 'epochs': EPOCHS})
    report(result)
    print('COMPLETE', json.dumps({'selected': selected, 'accuracy': tests[selected]['accuracy'],
                                'macro_f1': tests[selected]['macro_f1']}), flush=True)


def report(result):
    selected = result['selected_before_test']
    chosen = result['final_test'][selected]
    lines = ['# Movement classifier: first experiment', '',
             f'Selected approach: **{selected}**, chosen by five-fold subject-balanced macro-F1 before final testing.', '',
             '| Model | CV mean subject macro-F1 | Test accuracy | Test macro-F1 | Test balanced accuracy |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in ['cnn', 'extra_trees']:
        r = result['final_test'][name]
        lines.append(f"| {name} | {result['cv_selection_scores'][name]:.3f} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | {r['balanced_accuracy']:.1%} |")
    lines += ['', '| Class | Precision | Recall | F1 | Test windows |', '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        r = chosen['classification_report'][name]
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ['', '| Test subject | Accuracy | Macro-F1 |', '| --- | ---: | ---: |']
    for subject, r in chosen['subjects'].items():
        lines.append(f"| {subject} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} |")
    lines += ['', '## Scope and limitations', '',
              'Uses 18 IMU channels from foot, shank and thigh, with 256-frame windows at nominal 100 Hz. '
              'Walking portions are merged into one class. Only full windows are used; joint angles, subject IDs, '
              'trial names and label metadata are not model inputs. The CNN is trained from scratch, not from '
              'BioMAT weights that have seen cross-validation subjects.', '',
              'The existing dataset was presegmented using known activity labels. Transition portions are grouped '
              'with related ramp/stair activities; standing, idle, turns and running are excluded. This measures '
              'classification of curated activity windows, not continuous real-world detection or transition timing. '
              'Existing target-based quality exclusions are inherited from the regression preparation. '
              'Overlapping windows remain within a subject partition, but are not independent observations.', '',
              'Test subjects AB23/AB24/AB25 were not used to train or select the classifier. They were previously '
              'used in the separate regression experiment. AB27/AB28/AB30 remain unused. '
              'Phone recordings, sensor orientation/units and device transfer are not validated. '
              'Five-way scores are uncalibrated; a confident label does not establish that an unfamiliar activity is supported.', '',
              'Artifacts: `Camargo et al. Dataset/movement_classifier/v1/`. '
              'The selected model is recorded in `selection.json`; final models are in `final/`.', '',
              '[Confusion matrix](movement_classifier_confusion.png)', '']
    (ROOT / 'reports/movement_classifier_results.md').write_text('\n'.join(lines))
    cm = np.array(chosen['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    labels = [s.replace('_', '\n') for s in CLASSES]
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    ax.set(xlabel='Predicted activity', ylabel='True activity', title=f'{selected}: held-out activity classification')
    for i in range(5):
        for j in range(5):
            fraction = cm[i, j] / cm[i].sum()
            ax.text(j, i, f'{cm[i,j]}\n{fraction:.0%}', ha='center', va='center',
                    color='white' if fraction > 0.55 else 'black')
    fig.colorbar(im, ax=ax, label='Fraction within each true activity')
    fig.savefig(ROOT / 'reports/movement_classifier_confusion.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

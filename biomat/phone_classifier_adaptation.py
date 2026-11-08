"""Fine-tune the 100 Hz classifier with whole-trial phone holdouts and Camargo replay."""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch import nn

from .classifier import CLASSES, score
from .prepare import ROOT, TRAIN, TEST
from .two_sensor_100hz import OUT as BASE, load, new_model, predict, replace_checkpoint
from .train import write_json

OUT = BASE / 'phone_adaptation_v1'
PHONE_REPORT = ROOT / 'reports/phone_100hz_classifier'
REPORT = ROOT / 'reports/phone_classifier_adaptation_v1'
EPOCHS, HALF_BATCH, LR, SEED = 20, 32, 0.0001, 42
PHONE_LABELS = [0, 3, 4]


def phone_data():
    report = json.loads((PHONE_REPORT / 'summary.json').read_text())
    audit = json.loads((ROOT / 'reports/phone_100hz_audit/audit.json').read_text())
    source = Path(audit['source_directory'])
    for name, sha in report['protocol']['source_sha256'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Phone source changed since inference: ' + name)
    arrays = [[] for _ in range(6)]
    provenance = {}
    for trial in report['trials']:
        if trial['status'] != 'evaluated':
            continue
        key = f"{trial['activity']}_trial_{trial['trial']}"
        path = PHONE_REPORT / key / 'predictions.npz'
        provenance[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as a:
            keep = a['usable']
            n = int(keep.sum())
            values = [a['canonical_input'][keep], np.full(n, CLASSES.index(trial['activity']), dtype=np.int64),
                      np.full(n, trial['trial'], dtype=np.int64), np.full(n, key),
                      a['probabilities'][keep], a['start_s'][keep]]
            for destination, value in zip(arrays, values):
                destination.append(value)
    return (*[np.concatenate(a) for a in arrays], provenance)


def fold_masks(groups):
    """All activities with the same trial number are held out together."""
    for group in sorted(np.unique(groups)):
        held_out = groups == group
        yield int(group), ~held_out, held_out


def metrics(y, probabilities, trials):
    predicted = probabilities.argmax(1)
    records = {}
    for trial in sorted(np.unique(trials)):
        mask = trials == trial
        expected = int(y[mask][0])
        assert np.all(y[mask] == expected)
        counts = np.bincount(predicted[mask], minlength=5)
        winners = np.flatnonzero(counts == counts.max())
        dominant = CLASSES[winners[0]] if len(winners) == 1 else 'tie'
        records[str(trial)] = {'activity': CLASSES[expected], 'windows': int(mask.sum()),
                               'matching_windows': int((predicted[mask] == expected).sum()),
                               'agreement': float(np.mean(predicted[mask] == expected)),
                               'dominant_prediction': dominant,
                               'dominant_matches': dominant == CLASSES[expected]}
    return {'windows': len(y), 'matching_windows': int((predicted == y).sum()),
            'agreement': float(np.mean(predicted == y)),
            'macro_f1_over_three_recorded_activities': float(f1_score(y, predicted, labels=PHONE_LABELS, average='macro', zero_division=0)),
            'confusion_matrix_all_five_classes': confusion_matrix(y, predicted, labels=range(5)).tolist(),
            'by_activity': {CLASSES[label]: {'windows': int((y == label).sum()),
                                           'matching_windows': int(((y == label) & (predicted == label)).sum()),
                                           'agreement': float(np.mean(predicted[y == label] == label))}
                            for label in PHONE_LABELS if np.any(y == label)},
            'trials_with_matching_dominant_prediction': sum(r['dominant_matches'] for r in records.values()),
            'trials': records}


def balanced_indices(labels, rng, count):
    classes = np.unique(labels)
    chosen = rng.choice(classes, size=count, replace=True)
    return np.array([rng.choice(np.flatnonzero(labels == label)) for label in chosen])


def fit(saved, phone_x, phone_y, lab_x, lab_y, folder, training_trials, protocol, device):
    folder.mkdir(parents=True, exist_ok=True)
    training_trials = sorted(str(trial) for trial in training_trials)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = new_model('classifier').to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    mean, scale = saved['mean'].numpy(), saved['scale'].numpy()
    px = torch.from_numpy((phone_x - mean) / scale)
    lx = torch.from_numpy((lab_x - mean) / scale)
    py, ly = torch.from_numpy(phone_y), torch.from_numpy(lab_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    steps = max(4, math.ceil(len(phone_y) / HALF_BATCH))
    history = []
    print(folder.name, 'training phone windows:', len(phone_y), 'steps/epoch:', steps, flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Keep running statistics from the Camargo checkpoint; affine BN weights
        # and all other network parameters remain trainable.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        loss_sum = 0.
        for _ in range(steps):
            pi = balanced_indices(phone_y, rng, HALF_BATCH)
            li = balanced_indices(lab_y, rng, HALF_BATCH)
            x = torch.cat([px[pi], lx[li]]).to(device)
            y = torch.cat([py[pi], ly[li]]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite fine-tuning loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        row = {'epoch': epoch, 'mixed_training_loss': loss_sum / steps}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(folder.name, json.dumps(row), flush=True)
    checkpoint = {**saved, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  'epoch': EPOCHS, 'pretrained_epoch': saved['epoch'], 'adaptation_protocol': protocol,
                  'phone_training_trials': sorted(training_trials), 'phone_training_windows': len(phone_y)}
    torch.save(checkpoint, folder / 'model.tmp')
    replace_checkpoint(folder / 'model.tmp', folder / 'model.pt')
    write_json(folder / 'training.json', {'epochs': history, 'steps_per_epoch': steps,
                                        'phone_training_trials': sorted(training_trials)})
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        print('This version is already complete:', OUT / 'results.json')
        return
    x, y, groups, trials, baseline, times, provenance = phone_data()
    base_checkpoint = BASE / 'final/classifier/model.pt'
    base_sha = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    saved = torch.load(base_checkpoint, weights_only=True, map_location='cpu')
    baseline_report = json.loads((PHONE_REPORT / 'summary.json').read_text())
    assert base_sha == baseline_report['protocol']['checkpoint_sha256']
    folds = []
    for group, training, held_out in fold_masks(groups):
        train_trials = sorted(np.unique(trials[training]).tolist())
        test_trials = sorted(np.unique(trials[held_out]).tolist())
        assert not set(train_trials) & set(test_trials)
        assert set(y[training]) == set(PHONE_LABELS)
        folds.append({'held_out_trial_number': group, 'train_trials': train_trials, 'held_out_trials': test_trials,
                      'train_windows': int(training.sum()), 'held_out_windows': int(held_out.sum())})
    protocol = {'version': 'phone_adaptation_v1', 'initial_checkpoint_sha256': base_sha,
                'epochs_fixed_before_evaluation': EPOCHS, 'learning_rate': LR, 'optimizer': 'AdamW',
                'weight_decay': 0.0001, 'batch_size': HALF_BATCH * 2,
                'batch_composition': '32 phone + 32 Camargo; each domain sampled uniformly by available class, then by window, with replacement',
                'steps_per_epoch': 'max(4, ceil(training_phone_windows/32))',
                'loss': 'Mean cross entropy on balanced 50:50 domain batch',
                'batchnorm': 'Frozen pretrained running statistics; affine parameters trainable; dropout remains active during training',
                'scaler': 'Unchanged Camargo training scaler; no phone/test refitting',
                'phone_preprocessing': 'Unchanged audited raw g/rad/s path, epoch alignment, 250-sample windows, 50-sample stride, existing gate',
                'labels': 'User-supplied trial-level activity on previously accepted windows; timed ground truth absent',
                'camargo_replay_subjects': TRAIN, 'camargo_retention_test_subjects': TEST,
                'folds': folds, 'seed': SEED,
                'validation_selection': 'No early stopping, threshold tuning, hyperparameter search or epoch selection on held-out phone trials',
                'final_model': 'Fresh fine-tuning of original Camargo checkpoint using all accepted phone windows plus Camargo training replay',
                'evaluation_scope': 'Exploratory trial-held-out adaptation assessment on previously inspected recordings; no participant IDs or independent new session test. Final checkpoint uses all recordings and needs fresh phone testing.',
                'phone_cache_sha256': provenance}
    if (OUT / 'protocol.json').exists() and json.loads((OUT / 'protocol.json').read_text()) != protocol:
        raise ValueError('Protocol changed; use a separate experiment version')
    write_json(OUT / 'protocol.json', protocol)
    write_json(REPORT / 'protocol.json', protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, '| whole-trial folds:', len(folds), '| accepted phone windows:', len(y), flush=True)
    lab_x, _, lab_y, _, _ = load(TRAIN)
    out_of_fold = np.full((len(x), 5), np.nan, dtype=np.float32)
    assigned = np.zeros(len(x), dtype=int)
    fold_results = []
    for group, training, held_out in fold_masks(groups):
        folder = OUT / f'heldout_trial_{group}'
        model = fit(saved, x[training], y[training], lab_x, lab_y, folder, np.unique(trials[training]), protocol, device)
        prediction = predict(model, x[held_out], saved['mean'].numpy(), saved['scale'].numpy(), device, 'classifier')
        out_of_fold[held_out] = prediction
        assigned[held_out] += 1
        result = {'held_out_trial_number': group, 'baseline': metrics(y[held_out], baseline[held_out], trials[held_out]),
                  'adapted': metrics(y[held_out], prediction, trials[held_out])}
        write_json(folder / 'metrics.json', result)
        fold_results.append(result)
        print('HELD-OUT', group, 'baseline', result['baseline']['agreement'], 'adapted', result['adapted']['agreement'], flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    assert np.all(assigned == 1) and np.isfinite(out_of_fold).all()
    np.savez(OUT / 'out_of_fold_predictions.npz', probabilities=out_of_fold, baseline=baseline,
             labels=y, trial_number=groups, trial_name=trials, start_s=times, canonical_input=x)
    model = fit(saved, x, y, lab_x, lab_y, OUT / 'final', np.unique(trials), protocol, device)
    tx, ty, tl, ta, ts = load(TEST)
    lab_prediction = predict(model, tx, saved['mean'].numpy(), saved['scale'].numpy(), device, 'classifier')
    np.savez(OUT / 'camargo_retention_predictions.npz', probabilities=lab_prediction, labels=tl, subjects=ts)
    previous_lab = json.loads((BASE / 'results_classifier.json').read_text())['test']
    result = {'protocol': protocol, 'phone_baseline': metrics(y, baseline, trials),
              'phone_out_of_fold': metrics(y, out_of_fold, trials), 'folds': fold_results,
              'camargo_retention_baseline': previous_lab, 'camargo_retention_after_adaptation': score(tl, lab_prediction, ts),
              'final_checkpoint': str((OUT / 'final/model.pt').relative_to(ROOT)),
              'final_checkpoint_sha256': hashlib.sha256((OUT / 'final/model.pt').read_bytes()).hexdigest()}
    write_json(OUT / 'results.json', result)
    write_json(REPORT / 'results.json', result)
    table = pd.DataFrame({'trial': trials, 'trial_number': groups, 'start_s_from_overlap': times,
                          'trial_label': np.array(CLASSES)[y], 'baseline_prediction': np.array(CLASSES)[baseline.argmax(1)],
                          'held_out_prediction': np.array(CLASSES)[out_of_fold.argmax(1)]})
    for i, name in enumerate(CLASSES):
        table[name + '_held_out_score'] = out_of_fold[:, i]
    table.to_csv(REPORT / 'out_of_fold_windows.csv', index=False)
    before, after = result['phone_baseline'], result['phone_out_of_fold']
    lines = ['# Phone adaptation of the 100 Hz activity classifier', '',
             f"Trial-held-out phone label agreement: **{before['matching_windows']}/{before['windows']} ({before['agreement']:.1%}) before**, **{after['matching_windows']}/{after['windows']} ({after['agreement']:.1%}) after**.",
             'Every adapted prediction in this comparison comes from a fold that excluded the entire recording and all other activities with its trial number. Overlapping windows were never split across training and evaluation.', '',
             '| Recorded activity | Before | Trial-held-out after adaptation |', '| --- | ---: | ---: |']
    for label in PHONE_LABELS:
        name = CLASSES[label]
        a, b = before['by_activity'][name], after['by_activity'][name]
        lines.append(f"| {name} | {a['matching_windows']}/{a['windows']} ({a['agreement']:.1%}) | {b['matching_windows']}/{b['windows']} ({b['agreement']:.1%}) |")
    lines += ['', '| Recording | Before | Held-out after | Dominant after |', '| --- | ---: | ---: | --- |']
    for name, value in after['trials'].items():
        a = before['trials'][name]
        lines.append(f"| {name} | {a['matching_windows']}/{a['windows']} | {value['matching_windows']}/{value['windows']} | {value['dominant_prediction']} |")
    lines += ['', '## Camargo retention check', '',
              f"On the previously used AB23/24/25 test participants: {previous_lab['accuracy']:.2%} before and {result['camargo_retention_after_adaptation']['accuracy']:.2%} after final adaptation. The replay uses AB06–AB21 only. This is a retention check on previously evaluated data, not a fresh test study.", '',
              '## Training and limits', '',
              'Each fold starts from the original 100 Hz classifier. Fine-tuning uses 20 fixed epochs, AdamW at 0.0001, batches of 32 class-balanced phone and 32 class-balanced Camargo windows. BatchNorm running statistics, input scaler, sensor preprocessing and motion gate remain fixed. All five output classes remain available.',
              'Only the 182 previously accepted windows from eight complete recordings are used. The other 215 windows remain outside evaluation; walking trial 1 lacks phone 2. Trial-level labels may include transitions or other movements. Clock synchronization remains unverified. All these recordings were already inspected; this is an exploratory adaptation result, not an untouched final evaluation. Participant identities were not supplied, so cross-person generalization cannot be assessed.',
              'The final checkpoint is independently fine-tuned on ALL accepted phone recordings plus Camargo replay. Its performance must next be tested on NEW recordings. Do not score that final checkpoint on these same recordings and call the result generalization.',
              'BioMAT angle-regression weights were not changed. The existing shared ZIP still contains the earlier classifier.', '',
              f"Final checkpoint: `{result['final_checkpoint']}`.",
              f"SHA-256: `{result['final_checkpoint_sha256']}`.", '',
              '[Per-window held-out predictions](out_of_fold_windows.csv) · [Full results](results.json) · [Fixed protocol](protocol.json)', '']
    (REPORT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    assert hashlib.sha256(base_checkpoint.read_bytes()).hexdigest() == base_sha
    print('COMPLETE', json.dumps({'phone_before': before['agreement'], 'phone_out_of_fold_after': after['agreement'],
                                'camargo_before': previous_lab['accuracy'], 'camargo_after': result['camargo_retention_after_adaptation']['accuracy'],
                                'checkpoint': result['final_checkpoint']}), flush=True)


if __name__ == '__main__':
    main()

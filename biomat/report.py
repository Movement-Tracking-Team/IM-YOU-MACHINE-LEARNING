"""Verify completed prediction artifacts and summarize the fixed baseline."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .prepare import ROOT, TRAIN, TEST
from .train import RUNS, metrics


def main():
    run = RUNS / 'paper_standard_v1'
    reports = ROOT / 'reports'
    names = [f'fold_{i}' for i in range(1, 6)] + ['final']
    results, histories = {}, {}
    for name in names:
        folder = run / name
        if not (folder / 'metrics.json').exists():
            raise RuntimeError(f'{name} is not complete; no final report generated')
        status = json.loads((folder / 'status.json').read_text())
        protocol = json.loads((folder / 'protocol.json').read_text())
        assert status['state'] == 'complete' and status['epochs'] == 50
        assert not set(protocol['train_subjects']) & set(protocol['evaluation_subjects'])
        if name == 'final':
            assert protocol['train_subjects'] == TRAIN and protocol['evaluation_subjects'] == TEST
        with np.load(folder / 'predictions.npz', allow_pickle=False) as a:
            assert a['predictions'].shape == a['targets'].shape
            assert np.isfinite(a['predictions']).all() and np.isfinite(a['targets']).all()
            assert set(a['subjects']) == set(protocol['evaluation_subjects'])
            computed = metrics(a['predictions'], a['targets'], a['valid'], a['activities'], a['subjects'])
        recorded = json.loads((folder / 'metrics.json').read_text())
        np.testing.assert_allclose(computed['pooled_rmse_degrees'], recorded['pooled_rmse_degrees'], atol=1e-9)
        results[name] = computed
        histories[name] = [json.loads(s) for s in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert {r['epoch'] for r in histories[name]} == set(range(1, 51))

    final = results['final']
    fold_scores = [results[n]['pooled_mean_rmse_degrees'] for n in names[:-1]]
    by_activity = {}
    for activity in sorted({r['activity'] for r in final['by_subject_activity']}):
        by_activity[activity] = np.mean([r['rmse_degrees'] for r in final['by_subject_activity']
                                        if r['activity'] == activity], axis=0).tolist()
    lines = [
        '# BioMAT baseline results', '',
        'All five cross-validation folds and the final model completed 50 epochs. '
        'Saved prediction metrics were recomputed and verified; subject partitions were checked.', '',
        '## Final held-out test', '',
        'Training subjects: AB06–AB21. Test subjects: AB23, AB24, AB25. '
        'The final holdout was evaluated after epoch 50, without selecting a checkpoint on it.', '',
        '| Joint | Pooled RMSE (degrees) | Valid frames only |', '| --- | ---: | ---: |']
    for joint, pooled, masked in zip(['Hip', 'Knee', 'Ankle'], final['pooled_rmse_degrees'], final['valid_frames_rmse_degrees']):
        lines.append(f'| {joint} | {pooled:.3f} | {masked:.3f} |')
    lines += [f"| Mean across joints | {final['pooled_mean_rmse_degrees']:.3f} | {np.mean(final['valid_frames_rmse_degrees']):.3f} |", '',
              f"Equal-weight subject/activity mean RMSE: **{final['subject_activity_macro_rmse_degrees']:.3f} degrees**. "
              f"Population standard deviation across the three subject means: {final['subject_std_degrees']:.3f} degrees. "
              'This is not a confidence interval.', '',
              '| Test subject | Mean across activities and joints (degrees) |', '| --- | ---: |']
    for subject, value in final['subject_mean_rmse_degrees'].items():
        lines.append(f'| {subject} | {value:.3f} |')
    lines += ['', '| Activity | Hip | Knee | Ankle | Mean |', '| --- | ---: | ---: | ---: | ---: |']
    for activity, values in by_activity.items():
        lines.append('| ' + activity + ' | ' + ' | '.join(f'{v:.3f}' for v in values + [float(np.mean(values))]) + ' |')
    lines += ['', 'Activity values above average subject-specific RMSEs equally.', '',
              '## Cross-validation', '', '| Fold | Epoch-50 pooled mean RMSE | Lowest logged validation RMSE | Epoch of lowest |',
              '| --- | ---: | ---: | ---: |']
    for name in names[:-1]:
        best = min(histories[name], key=lambda x: x['validation_pooled_rmse'])
        lines.append(f"| {name} | {results[name]['pooled_mean_rmse_degrees']:.3f} | {best['validation_pooled_rmse']:.3f} | {best['epoch']} |")
    lines += ['', f'Mean of the five epoch-50 fold scores: **{statistics.mean(fold_scores):.3f} degrees** '
              f'(population SD across folds {statistics.pstdev(fold_scores):.3f}). '
              'Each fold contains different validation subjects; this is not the final test score.', '',
              'The lowest logged values are diagnostic only. Earlier model checkpoints were not retained, '
              'and these values do not replace the preselected final-epoch baseline.', '',
              '## Interpretation and limits', '',
              'The paper reports 5.5 degrees mean test RMSE. Compare that reference cautiously: '
              'this run uses the paper architecture/settings with corrected standardization and modern PyTorch, '
              'but has 12,047 windows versus 12,092 reported in the paper. '
              'The exact published aggregation and processed-dataset equivalence are not established. '
              'See [protocol](reproduction_protocol.md) for all deviations. '
              'A completed run does not establish an exact reproduction.', '',
              'Validation performance varies by subject and the lowest logged validation errors occur before epoch 50. '
              'Validation-based checkpointing, training stability and preprocessing parity are appropriate next investigations. '
              'Do not choose changes based on the now-observed final test results; keep AB27/AB28/AB30 reserved for a fresh evaluation.', '',
              'The authors\' downloaded BioMAT.pt was not loaded or run in this experiment.', '',
              '## Files', '',
              '- Final model: `Camargo et al. Dataset/biomat_reproduction/runs/paper_standard_v1/final/model.pt`.',
              '- Detailed metrics and predictions remain in that same ignored run directory.',
              '- [Learning curves](baseline_learning_curves.png).',
              '- [Held-out test errors](baseline_test_errors.png).', '',
              'Reference: [BioMAT paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346710/).', '']
    (reports / 'baseline_results.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for name, ax in zip(names, axes.flat):
        history = histories[name]
        ax.plot([r['epoch'] for r in history], [np.sqrt(r['train_mse']) for r in history], label='Train sqrt(MSE)')
        if name != 'final':
            ax.plot([r['epoch'] for r in history], [r['validation_pooled_rmse'] for r in history], label='Validation mean joint RMSE')
        ax.set(title=name.replace('_', ' ').title(), xlabel='Epoch', ylabel='Degrees')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('BioMAT baseline: training and validation histories\nTraining sqrt(MSE) and mean joint RMSE use different aggregation rules')
    fig.savefig(reports / 'baseline_learning_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(['Hip', 'Knee', 'Ankle'], final['pooled_rmse_degrees'], color='#286f9b')
    axes[0].set(title='Final test: pooled joint errors', ylabel='RMSE (degrees)')
    axes[1].bar(list(final['subject_mean_rmse_degrees']), list(final['subject_mean_rmse_degrees'].values()), color='#389078')
    axes[1].set(title='Final test: subject means', ylabel='Mean activity/joint RMSE (degrees)')
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.2)
    fig.savefig(reports / 'baseline_test_errors.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'final_pooled_mean_rmse': final['pooled_mean_rmse_degrees'],
                      'final_subject_activity_macro_rmse': final['subject_activity_macro_rmse_degrees'],
                      'fold_mean_rmse': statistics.mean(fold_scores), 'report': str(reports / 'baseline_results.md')}, indent=2))


if __name__ == '__main__':
    main()

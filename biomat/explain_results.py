"""Visual explanations from saved test predictions; does not run any model."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from .prepare import DATA, CACHE


def main():
    run = DATA / 'biomat_reproduction/runs/paper_standard_v1/final'
    out = DATA / 'biomat_reproduction/visualizations'
    out.mkdir(parents=True, exist_ok=True)
    with np.load(run / 'predictions.npz', allow_pickle=False) as a:
        pred, target = a['predictions'], a['targets']
        valid = np.arange(256)[None, :] < a['valid'][:, None]
        errors = np.abs(pred.astype(np.float64) - target)[valid]
        # Fixed selection independent of prediction quality: first full walking window.
        idx = int(np.flatnonzero((a['subjects'] == 'AB23') &
                                (a['activities'] == 'levelground1') & (a['valid'] == 256))[0])
        subject_index = int(np.count_nonzero(a['subjects'][:idx] == 'AB23'))
        source = json.loads((CACHE / 'AB23.json').read_text())['metadata'][subject_index]
        stats = {'selection': 'First complete levelground1 window of AB23; not selected by error',
                 'source': source, 'prediction_index': idx,
                 'mean_absolute_error_degrees': float(errors.mean()),
                 'per_joint_mae': errors.mean(axis=0).tolist(),
                 'fraction_within_5_degrees': float((errors <= 5).mean()),
                 'fraction_within_10_degrees': float((errors <= 10).mean()),
                 'counting': 'All valid joint/time predictions; overlapping windows retain repeated frames'}
        (out / 'explanation_statistics.json').write_text(json.dumps(stats, indent=2))
        fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
        colors = ('#196da8', '#e77b24')
        time = np.arange(256) / 100
        for j, (ax, name) in enumerate(zip(axes, ['Hip', 'Knee', 'Ankle'])):
            ax.plot(time, target[idx, :, j], color=colors[0], linewidth=2.2, label='Reference angle (dataset)')
            ax.plot(time, pred[idx, :, j], color=colors[1], linewidth=2, linestyle='--', label='Our model prediction')
            ax.fill_between(time, target[idx, :, j], pred[idx, :, j], color=colors[1], alpha=0.13)
            rmse = np.sqrt(np.mean((pred[idx, :, j] - target[idx, :, j]) ** 2))
            ax.set_ylabel(f'{name} angle (degrees)')
            ax.set_title(f'{name}: this window RMSE = {rmse:.2f} degrees', loc='left', fontsize=11)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 2.55)
        axes[0].legend(loc='upper right', fontsize=9)
        axes[-1].set_xlabel('Time within the 2.56-second window (seconds)')
        fig.suptitle('What our model predicted on a person it never trained on\n'
                     'AB23: first complete level-walking window; shaded gap is prediction error', fontsize=14)
        fig.supxlabel('Negative angles follow the dataset coordinate convention; they are not negative errors.', fontsize=9)
        fig.savefig(out / 'actual_vs_predicted.png', dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 0], [0, 1.05], color='#343a40', linewidth=9, solid_capstyle='round')
    ax.plot([0, 0], [0, -0.95], ':', color='#b0b5ba')
    for angle, color, style in [(60, '#196da8', '-'), (65.5, '#e77b24', '--')]:
        theta = np.radians(angle)
        ax.plot([0, np.sin(theta)], [0, -np.cos(theta)], style, color=color, linewidth=5,
                label=f'{"Reference" if angle == 60 else "Prediction"}: {angle:g}°')
    ax.add_patch(Arc((0, 0), 0.75, 0.75, theta1=-90, theta2=-30, color='#196da8', linewidth=2))
    ax.scatter([0], [0], s=120, c='#343a40', zorder=5)
    ax.text(-0.17, 0.6, 'Thigh', ha='right', fontsize=12)
    ax.text(0.65, -0.8, 'Lower leg', ha='center', fontsize=12)
    ax.text(0.32, -1.12, '60° reference → 65.5° prediction\nError at this instant = 5.5°', ha='center', fontsize=12)
    ax.set(xlim=(-0.65, 1.35), ylim=(-1.4, 1.25), aspect='equal')
    ax.axis('off')
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.set_title('What a 5.5° error looks like\nIllustrative values, not a recorded sample', fontsize=12)
    ax = axes[1]
    labels = ['Within 5°', 'Within 10°']
    values = [100 * stats['fraction_within_5_degrees'], 100 * stats['fraction_within_10_degrees']]
    bars = ax.barh(labels, values, color=['#389078', '#196da8'], height=0.5)
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=5, fontsize=13)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Percentage of evaluated joint/time predictions')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    ax.set_title('Our actual held-out test errors\nPadding excluded; overlapping frames counted', fontsize=12)
    fig.savefig(out / 'angle_error_explained.png', dpi=170)
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

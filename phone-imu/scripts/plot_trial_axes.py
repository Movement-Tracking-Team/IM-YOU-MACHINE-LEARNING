"""Plot signed XYZ measurements for thigh and shank on the shared session timeline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def overlay_figure(groups, report, prefix, title, units, colors, segment=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(top=.77, bottom=.16, left=.08, right=.985)
    heading = f'{segment.upper()} — {title}' if segment else f'{title} — X/Y/Z overlaid for both phones'
    subtitle = ('Phone 1' if segment == 'Thigh' else 'Phone 2') + ' • X, Y and Z on one graph' if segment else 'Phone 1: thigh (solid)  •  Phone 2: shank (dashed)'
    fig.suptitle(heading, x=.08, y=.97, ha='left', fontsize=18)
    fig.text(.08, .915, subtitle, color='#52616b')
    labels = [segment] if segment else ['Thigh', 'Shank']
    for group, body_segment, style in zip(groups, labels, ['-', '--']):
        if group.session_id.nunique() != 1 or group.session_id.iloc[0] != report['session_id']:
            raise ValueError('Input file does not match the analysed session')
        for axis, color in zip('xyz', colors):
            if not np.isfinite(group[['session_time_s', prefix + axis]].to_numpy()).all():
                raise ValueError('Nonfinite time or measurement')
            ax.plot(group.session_time_s, group[prefix + axis], color=color, ls=style,
                    lw=1.15, alpha=.85, label=axis.upper() if segment else f'{body_segment} — {axis.upper()}')
    start, stop = report['walking_estimate_start_s'], report['walking_estimate_stop_s']
    ax.axvspan(start, stop, color='#389d73', alpha=.07, zorder=0)
    for boundary in [start, stop]:
        ax.axvline(boundary, color='#5d6a71', ls=':', lw=1)
    ax.axhline(0, color='#87929a', lw=.7, zorder=0)
    ax.set_xlim(0, report['recording_duration_s'])
    ax.set_xlabel('Time since shared recording start (s)')
    ax.set_ylabel(f'{title} ({units})')
    ax.grid(color='#e3e7e9', lw=.6)
    ax.margins(y=.08)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.54, .885),
               ncol=3 if segment else 6, frameon=False, handlelength=3)
    fig.text(.08, .065, f'Shading: estimated walking interval {start:.1f}–{stop:.1f} s. Original signed measurements; no smoothing or time shifting.',
             fontsize=10, color='#52616b')
    fig.text(.08, .03, 'Axes are phone coordinates; anatomical-axis calibration has not been applied.', fontsize=10, color='#52616b')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_directory', type=Path)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument('--overlay', action='store_true', help='Put XYZ from both phones on one graph per sensor type')
    layout.add_argument('--separate-phones', action='store_true', help='One XYZ graph per phone and sensor type')
    args = parser.parse_args()
    root = args.analysis_directory
    report = json.loads((root / 'analysis.json').read_text(encoding='utf8'))
    groups = [pd.read_csv(root / f'{segment}_full.csv') for segment in ('thigh', 'shank')]
    start, stop = report['walking_estimate_start_s'], report['walking_estimate_stop_s']
    duration = report['recording_duration_s']
    colors = ['#b9423a', '#187b52', '#2869b0']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    for prefix, title, units, basename in [
        ('a', 'Acceleration including gravity', 'm/s²', 'acceleration_xyz'),
        ('g', 'Gyroscope angular velocity', 'deg/s', 'gyroscope_xyz'),
    ]:
        if args.separate_phones:
            for group, segment in zip(groups, ['Thigh', 'Shank']):
                fig = overlay_figure([group], report, prefix, title, units, colors, segment=segment)
                for extension in ('png', 'pdf'):
                    output = root / f'{segment.lower()}_{basename}.{extension}'
                    fig.savefig(output, dpi=180)
                    print(output.resolve())
                plt.close(fig)
            continue
        if args.overlay:
            fig = overlay_figure(groups, report, prefix, title, units, colors)
            for extension in ('png', 'pdf'):
                output = root / f'{basename}_overlay.{extension}'
                fig.savefig(output, dpi=180)
                print(output.resolve())
            plt.close(fig)
            continue
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey='row')
        fig.subplots_adjust(top=.84, bottom=.14, left=.085, right=.985, hspace=.23, wspace=.12)
        fig.suptitle(f'{title} — individual X, Y and Z axes', x=.085, y=.975, ha='left', fontsize=19)
        fig.text(.085, .927, f'Walking trial • {duration:.2f} s recording • approximately 60 Hz per phone', color='#52616b')
        for col, (group, label) in enumerate(zip(groups, ['Phone 1 — thigh', 'Phone 2 — shank'])):
            axes[0, col].set_title(label, fontsize=14, pad=14, fontweight='bold')
            if group.session_id.nunique() != 1 or group.session_id.iloc[0] != report['session_id']:
                raise ValueError('Input file does not match the analysed session')
            if not np.isfinite(group[['session_time_s', *[prefix + a for a in 'xyz']]].to_numpy()).all():
                raise ValueError('Nonfinite time or measurement')
            for row, axis in enumerate('xyz'):
                ax = axes[row, col]
                ax.axvspan(start, stop, color='#389d73', alpha=.09, zorder=0)
                ax.axvline(start, color='#5d6a71', ls='--', lw=.8)
                ax.axvline(stop, color='#5d6a71', ls='--', lw=.8)
                ax.axhline(0, color='#a8b0b5', lw=.7, zorder=0)
                ax.plot(group.session_time_s, group[prefix + axis], color=colors[row], lw=1.0)
                ax.set_xlim(0, duration)
                ax.grid(axis='both', color='#e3e7e9', lw=.6)
                ax.tick_params(labelsize=10)
                ax.text(.015, .92, f'{axis.upper()} axis', transform=ax.transAxes,
                        color=colors[row], fontweight='bold', va='top')
                if col == 0:
                    ax.set_ylabel(f'{prefix}{axis} ({units})')
                if row == 2:
                    ax.set_xlabel('Time since shared recording start (s)')
        # Identical limits for corresponding axes on the two phones, with no clipping.
        for row, axis in enumerate('xyz'):
            values = np.concatenate([group[prefix + axis].to_numpy() for group in groups])
            low, high = min(0, values.min()), max(0, values.max())
            pad = max((high - low) * .10, .1)
            axes[row, 0].set_ylim(low - pad, high + pad)
        fig.legend(handles=[Patch(facecolor='#389d73', alpha=.18,
                                  label=f'Estimated walking interval: {start:.1f}–{stop:.1f} s')],
                   loc='lower center', bbox_to_anchor=(.5, .052), frameon=False)
        fig.text(.085, .028, 'Signed phone-coordinate measurements; no smoothing, resampling or anatomical-axis calibration applied.',
                 fontsize=10, color='#52616b')
        for extension in ('png', 'pdf'):
            output = root / f'{basename}.{extension}'
            fig.savefig(output, dpi=180)
            print(output.resolve())
        plt.close(fig)


if __name__ == '__main__':
    main()

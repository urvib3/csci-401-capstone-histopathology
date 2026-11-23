"""Render presentation figures from saved results without rerunning experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'


def read(name):
    return json.loads((HERE / name).read_text())


def overview(evaluation):
    methods = evaluation['methods']
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    choices = ['raw_tp', 'vote', 'global_type_set', 'breast_type_set']
    names = ['Raw HoVer-Net', 'Fixed-mask vote', 'Global WCSP', 'Breast WCSP']
    colors = ['#7d8e9f', '#879fb4', '#4b789e', '#158367']
    values = [methods[name]['bio_miou'] for name in choices]
    axis = axes[0]
    axis.barh(names, values, color=colors, height=.6)
    axis.invert_yaxis()
    axis.set_xlim(0, .52)
    axis.set_xlabel('Pooled five-class biological mIoU')
    axis.set_title('Same 169 Breast test patches', loc='left', fontweight='bold')
    for position, value in enumerate(values):
        axis.text(value + .007, position, f'{value:.5f}', va='center', fontsize=11)
    axis.spines[['top', 'right']].set_visible(False)
    axis = axes[1]
    contrasts = evaluation['results']['type_set']['comparisons']
    labels = [('raw_tp', 'Raw HoVer-Net', 'ci95'),
              ('matched_control', 'Fixed-mask vote', 'holm_ci95'),
              ('global', 'Global WCSP', 'ci95')]
    for position, (key, name, interval) in enumerate(labels):
        row = contrasts[key]
        lower, upper = row[interval]
        axis.hlines(position, lower, upper, color='#158367', linewidth=3)
        axis.plot(row['delta'], position, 'o', color='#158367', markersize=7)
    axis.set_yticks(range(3), [name for key, name, interval in labels])
    axis.set_ylim(2.6, -.6)
    axis.axvline(0, linestyle='--', color='#8b939b', linewidth=1)
    axis.set_xlim(-.002, .028)
    axis.set_xlabel('Breast WCSP − comparison method (mIoU)')
    axis.set_title('Improvement and uncertainty', loc='left', fontweight='bold')
    axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Organ-specific fitting + tuning helped patch-wide type-set WCSP', fontsize=14)
    figure.supxlabel('5,000 paired bootstrap draws · voting interval uses eight-family Holm inversion; other intervals unadjusted', fontsize=9)
    figure.savefig(FIGURES / 'breast_overview.png', dpi=170)
    plt.close(figure)


def classes(evaluation):
    methods = evaluation['methods']
    names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    positions = np.arange(5)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    raw = np.array(methods['raw_tp']['per_class_iou'])
    global_values = np.array(methods['global_type_set']['per_class_iou'])
    breast = np.array(methods['breast_type_set']['per_class_iou'])
    for offset, values, color, label in [(-.25, raw, '#8798a9', 'Raw HoVer-Net'),
                                         (0, global_values, '#4b789e', 'Global WCSP'),
                                         (.25, breast, '#158367', 'Breast WCSP')]:
        axes[0].bar(positions + offset, values, width=.23, color=color, label=label)
    axes[0].set_xticks(positions, names, rotation=20, ha='right')
    axes[0].set_ylim(0, .82)
    axes[0].set_ylabel('Per-class pooled IoU')
    axes[0].set_title('Where the gain appears', loc='left', fontweight='bold')
    axes[0].legend(fontsize=8, frameon=False, ncol=1)
    differences = breast - global_values
    axes[1].barh(names, differences, color=['#158367' if value >= 0 else '#b35e38' for value in differences])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='#8b939b', linewidth=1)
    axes[1].set_xlim(-.012, .06)
    axes[1].set_xlabel('Breast WCSP − global WCSP (IoU)')
    axes[1].set_title('Largest numerical gain: epithelial', loc='left', fontweight='bold')
    for position, value in enumerate(differences):
        alignment = 'left' if value >= 0 else 'right'
        offset = .001 if value >= 0 else -.001
        axes[1].text(value + offset, position, f'{value:+.5f}', va='center', ha=alignment, fontsize=10)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    figure.supxlabel('Descriptive class effects, no per-class significance claim · no Dead nuclei in this Breast cohort', fontsize=10)
    figure.savefig(FIGURES / 'breast_classes.png', dpi=170)
    plt.close(figure)


def main():
    FIGURES.mkdir(exist_ok=True)
    evaluation = read('evaluation.json')
    overview(evaluation)
    classes(evaluation)
    print('Rendered Breast overview and class comparison from saved evaluation.json.')


if __name__ == '__main__':
    main()

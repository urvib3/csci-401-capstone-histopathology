"""Reproduce D5 PanNuke figures from checked-in image_00748 arrays.

Run: MPLBACKEND=Agg python experiments/pannuke_d5/make_figures.py
The saved package contains label maps, not the original RGB image.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ['#eeeeee', '#df5555', '#66b86d', '#628ed4', '#b67bc4', '#e6b84b']
NAMES = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Non-neoplastic epithelial']
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-.5, 6), CMAP.N)


def main():
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    package = np.load(HERE / 'pannuke_preparation/pannuke_d5_patch_package.npz')
    initial, gt, active = (package[k] for k in ['predicted_labels', 'ground_truth_labels', 'active_mask'])
    refined = np.load(HERE / 'pannuke_refinement/wcsp_weak_labels.npy')
    changed = active & (initial != refined)
    corrected = changed & (initial != gt) & (refined == gt) & (gt > 0)
    new_errors = changed & (initial == gt) & (refined != gt) & (gt > 0)
    change_map = np.zeros((*initial.shape, 3)) + .94
    change_map[changed] = [0.95, .66, .17]
    change_map[corrected] = [.2, .7, .3]
    change_map[new_errors] = [.85, .15, .2]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.1))
    for ax, title, data, categorical in zip(axes,
        ['HoVer-Net class map', 'Active WCSP variables', 'Ground truth', 'WCSP, lambda = 1', f'Changes: {changed.sum()} pixels'],
        [initial, active, gt, refined, change_map], [True, False, True, True, False]):
        ax.imshow(data, cmap=CMAP if categorical else 'gray', norm=NORM if categorical else None, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.suptitle('PanNuke image_00748: preparation and refinement (164 x 164 label maps)', fontsize=14)
    fig.legend(handles=[Patch(color=c, label=n) for c,n in zip(COLORS,NAMES)], loc='lower center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0,.10,1,.92))
    fig.savefig(out / 'pipeline.png', dpi=160)
    plt.close(fig)

    # Center a close-up on an actual newly incorrect pixel, rather than inventing a helpful example.
    y, x = np.argwhere(new_errors)[0]
    y0, y1 = max(0,y-10), min(initial.shape[0],y+11)
    x0, x1 = max(0,x-10), min(initial.shape[1],x+11)
    fig, axes = plt.subplots(1,4,figsize=(12,4.2))
    for ax,title,data in zip(axes,['Ground truth','HoVer-Net','WCSP, lambda = 1','Changed pixels'],[gt,initial,refined,change_map]):
        ax.imshow(data[y0:y1,x0:x1], cmap=CMAP, norm=NORM if data.ndim==2 else None, interpolation='nearest')
        for yy,xx in np.argwhere(changed[y0:y1,x0:x1]):
            ax.add_patch(Rectangle((xx-.5,yy-.5),1,1,fill=False,edgecolor='black',lw=1))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'image_00748: local changes near (x={x}, y={y}); boxes mark changed pixels', fontsize=12)
    fig.legend(handles=[Patch(color=COLORS[2],label='Inflammatory'),Patch(color=COLORS[3],label='Connective'),Patch(color=COLORS[5],label='Non-neoplastic epithelial'),Patch(color='#d92633',label='Correct to wrong'),Patch(color='#f2a82b',label='Changed on GT background')],loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,.9))
    fig.savefig(out / 'example_hurts_zoom.png', dpi=160)
    plt.close(fig)
    print(f'Wrote figures: {changed.sum()} changes, {corrected.sum()} corrections, {new_errors.sum()} new errors.')


if __name__ == '__main__':
    main()

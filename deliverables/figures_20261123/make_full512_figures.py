"""Full-512 BCSS figures from portable result snapshots.

Run with the repository's plotting environment. Add --cache-root pointing to
the original experiment checkout to rebuild the qualitative example figure.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = REPO/'experiments/bcss_wcsp/results/full512_20261123'
BLUE, GREEN, GRAY, ORANGE = '#2a78d6', '#25876a', '#92989f', '#d46b39'
COLORS = np.array([[220,40,40],[40,160,40],[60,90,220],[240,200,30],[230,120,200]], dtype=np.uint8)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.spines.left':False, 'figure.facecolor':'white', 'savefig.facecolor':'white'})


def figures():
    data=json.loads((RESULTS/'corrections_dev.json').read_text())
    rows={r['name']:r for r in data['results'] if r['status']=='Complete'}
    names=[('Ordinary pixel WCSP','temp1_mu0'),('Ordinary hierarchy','temp1_mu1'),
           ('Temperature T=4 + hierarchy','temp4_mu1'),('No block fidelity + hierarchy','blockalpha0_mu1'),
           ('Presence gamma=4000 + hierarchy','presence4000_mu1'),('Connected-region moves','regions_components_mu0')]
    baseline=data['baseline']['mIoU']
    fig,axes=plt.subplots(1,2,figsize=(13.8,4.2),gridspec_kw={'width_ratios':[1.3,1]})
    for y,(label,name) in enumerate(names[::-1]):
        row=rows[name];value=row['metrics']['mIoU'];gain=100*(value-baseline)
        color=GREEN if name.startswith('regions') else BLUE if name.startswith('presence') else GRAY
        axes[0].barh(y,gain,color=color,height=.65)
        axes[0].text(gain+.025,y,f'{value:.5f}  (+{gain:.2f} pp)',va='center',fontsize=9)
    axes[0].set_yticks(range(len(names)),[x[0] for x in names[::-1]])
    axes[0].set_xlim(0,1.95);axes[0].set_xlabel('mIoU gain over raw U-Net (percentage points)')
    axes[0].set_title('Full-512 dev: 185 tiles, same U-Net baseline 0.65241',loc='left',fontsize=11)
    comparisons=names[2:]
    for y,(label,name) in enumerate(comparisons[::-1]):
        b=rows[name]['vs_control_bootstrap'];lo=100*b['ci_low'];hi=100*b['ci_high'];value=100*b['delta_mIoU']
        color=GREEN if name.startswith('regions') else BLUE
        axes[1].plot([lo,hi],[y,y],color=color,lw=2)
        axes[1].plot(value,y,'o',color=color,markersize=7)
        axes[1].text(hi+.035,y,f'{value:+.2f}',va='center',fontsize=9)
    axes[1].set_yticks(range(len(comparisons)),[x[0] for x in comparisons[::-1]])
    axes[1].axvline(0,color=GRAY,ls='--',lw=1)
    axes[1].set_xlim(-.25,2.6);axes[1].set_xlabel('mIoU points over matched control; 95% ROI interval')
    axes[1].set_title('Paired comparisons: 14 ROIs',loc='left',fontsize=11)
    for ax in axes:
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    fig.text(.01,.01,'Development-selected settings; exploratory intervals. Appearance grid remains incomplete in this snapshot.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'bcss_full512_dev.png',dpi=180,bbox_inches='tight');plt.close(fig)

    val=json.loads((RESULTS/'validation.json').read_text())
    provenance=json.loads((RESULTS/'provenance.json').read_text())
    methods=val['results'];base=provenance['protocols']['validation']['baseline']['mIoU']
    labels=['Raw full-resolution U-Net','Pixel WCSP, mu=0','Multiscale WCSP, mu=1']
    scores=[base]+[next(r['metrics']['mIoU'] for r in methods if r['mu']==mu) for mu in (0,1)]
    fig,ax=plt.subplots(figsize=(8.3,2.8))
    for y,(label,value) in enumerate(list(zip(labels,scores))[::-1]):
        ax.barh(y,100*(value-.665),left=66.5,color=BLUE if 'mu=1' in label else GRAY,height=.6)
        ax.text(100*value+.018,y,f'{100*value:.3f}%',va='center')
    ax.set_yticks(range(3),labels[::-1]);ax.set_xlim(66.5,67.65)
    ax.set_xlabel('mIoU (%) — axis starts at 66.5%');ax.tick_params(axis='y',length=0)
    ax.set_title('Full validation: 4,028 tiles accounted for; 3,229 scored; 45 ROIs',loc='left',fontsize=10)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT/'bcss_full512_validation.png',dpi=180,bbox_inches='tight');plt.close(fig)


def miou(confusion):
    diagonal=np.diag(confusion)
    union=confusion.sum(0)+confusion.sum(1)-diagonal
    return float(np.mean(diagonal[union>0]/union[union>0]))


def examples(cache):
    root=cache/'data/fullres_corrections_20261122'
    protocol=json.loads((root/'protocol.json').read_text())
    def score(record):
        name=f"{record['index']:04d}.npz"
        with np.load(root/'runs/temp1_mu0/dev/samples'/name) as z:
            a=miou(z['wcsp_confusion'])
        with np.load(root/'runs/regions_components_mu0/dev/samples'/name) as z:
            b=miou(z['wcsp_confusion'])
        return dict(index=record['index'],id=record['id'],control_tile_mIoU=a,
                    region_tile_mIoU=b,delta=b-a)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(score,protocol['samples']))
    positive=sorted((r for r in records if r['delta']>1e-12),key=lambda r:(r['delta'],r['index']))
    negative=sorted((r for r in records if r['delta']< -1e-12),key=lambda r:(r['delta'],r['index']))
    selected=[]
    for label,group in [('Median improved tile',positive),('Median worsened tile',negative)]:
        if group:selected.append(dict(group[len(group)//2],selection=label))
    fig,axes=plt.subplots(len(selected),6,figsize=(16,3.1*len(selected)),squeeze=False)
    for y,row in enumerate(selected):
        name=f"{row['index']:04d}.npz"
        with np.load(root/'assets'/name) as z:image=z['image']
        with np.load(cache/'data/fullres_multiscale_20261115/inputs'/name) as z:target=z['target']
        with np.load(root/'runs/temp1_mu0/dev/samples'/name) as z:raw,control=z['argmax'],z['wcsp']
        with np.load(root/'runs/regions_components_mu0/dev/samples'/name) as z:region=z['wcsp']
        valid=target!=255
        def colorize(labels):
            result=COLORS[np.minimum(labels,4)].copy();result[~valid]=235
            return result
        changed=valid&(control!=region)
        fixes=changed&(control!=target)&(region==target)
        breaks=changed&(control==target)&(region!=target)
        other=changed&(control!=target)&(region!=target)
        diff=np.full(image.shape,245,dtype=np.uint8)
        diff[fixes]=[40,160,70];diff[breaks]=[210,50,45];diff[other]=[235,155,40];diff[~valid]=210
        panels=[image,colorize(target),colorize(raw),colorize(control),colorize(region),diff]
        titles=['RGB','Ground truth','U-Net','Pixel WCSP','Region moves','Edits vs. pixel WCSP']
        for ax,panel,title in zip(axes[y],panels,titles):
            ax.imshow(panel);ax.set_title(title,fontsize=9);ax.axis('off')
        axes[y,0].set_ylabel(row['selection'],fontsize=9)
        axes[y,0].text(0,-.08,f"{row['selection']} — tile {row['index']}; delta tile mIoU {row['delta']:+.4f}",transform=axes[y,0].transAxes,fontsize=8)
        row.update(fixes_vs_control=int(fixes.sum()),breaks_vs_control=int(breaks.sum()),wrong_to_wrong_vs_control=int(other.sum()))
    legend=[Patch(color=COLORS[k]/255,label=label) for k,label in enumerate(['Tumor','Inflammatory','Stroma','Necrosis','Other'])]
    legend += [Patch(color='#28a046',label='Fix'),Patch(color='#d2322d',label='Break'),Patch(color='#eb9b28',label='Wrong to wrong')]
    fig.legend(handles=legend,loc='lower center',ncol=8,frameon=False,fontsize=8)
    fig.suptitle('Connected-region moves: median gain and median loss among changed dev tiles',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(OUT/'bcss_full512_region_examples.png',dpi=170,bbox_inches='tight');plt.close(fig)
    (RESULTS/'region_examples.json').write_text(json.dumps(dict(selection='Median positive and median negative tile-mIoU change vs matched pixel control; selected after inspection, illustrative only.',positive_tiles=len(positive),negative_tiles=len(negative),ties=len(records)-len(positive)-len(negative),examples=selected),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path)
    args=parser.parse_args()
    figures()
    if args.cache_root:examples(args.cache_root.resolve())

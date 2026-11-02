"""Visualize saved frozen outputs for Deliverable 5; no inference or optimization."""
import json,hashlib,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.build/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(HERE/'.build/cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

COLORS=['#202733','#d94747','#45a5e6','#64b967','#c993dc','#e8bf46']
NAMES=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic']
CHANGE=['#eeeeee','#278657','#c24d4d','#ed9e35']

def main():
    results=json.loads((HERE/'results.json').read_text())
    selection=json.loads((HERE/'selected_examples.json').read_text())
    examples=selection['examples'];assert len(examples)==2
    fig,axes=plt.subplots(2,5,figsize=(16,8.2))
    fig.subplots_adjust(left=.025,right=.99,top=.90,bottom=.20,hspace=.30,wspace=.035)
    fig.suptitle('PanNuke: raw type predictions and frozen pixel WCSP',fontsize=20,fontweight='bold',y=.98)
    for row,example in enumerate(examples):
        i=example['image_index'];z=np.load(HERE/f'predictions/patch_{i:04d}.npz');rgb=np.load(HERE/f'cache/rgb_{i:04d}.npy')[46:210,46:210]
        g,b,w=z['gt'],z['baseline'],z['wcsp'];changed=b!=w
        fix=(b!=g)&(w==g);brk=(b==g)&(w!=g);other=changed&(b!=g)&(w!=g)
        assert int(fix.sum())==example['fixes'] and int(brk.sum())==example['breaks']
        assert np.array_equal(changed,fix|brk|other)
        change=np.zeros(g.shape,np.uint8);change[fix]=1;change[brk]=2;change[other]=3
        for col,(array,title) in enumerate(zip([rgb,g,b,w,change],['Histology (center crop)','GT (evaluation only)','Raw TP argmax','WCSP: beta = 0.30','Changed pixels'])):
            ax=axes[row,col]
            if col==0:ax.imshow(array)
            elif col==4:ax.imshow(array,cmap=ListedColormap(CHANGE),vmin=0,vmax=3,interpolation='nearest')
            else:ax.imshow(array,cmap=ListedColormap(COLORS),vmin=0,vmax=5,interpolation='nearest')
            ax.set_title(title,fontsize=15);ax.axis('off')
        caption=f"{example['kind'].capitalize()} example: Fold 2 / {i:04d}, {example['tissue']} | biological mIoU change {example['delta']:+.5f} | {int(fix.sum())} fixes, {int(brk.sum())} breaks, {int(other.sum())} other wrong changes"
        axes[row,0].text(0,-.12,caption,transform=axes[row,0].transAxes,fontsize=13,ha='left',va='top',clip_on=False)
    fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(COLORS,NAMES)],loc='lower center',bbox_to_anchor=(.5,.065),ncol=6,frameon=False,fontsize=12)
    fig.legend(handles=[Patch(color=CHANGE[k],label=n) for k,n in enumerate(['Unchanged','Wrong → correct','Correct → wrong','Wrong → different wrong'])],loc='lower center',bbox_to_anchor=(.5,.018),ncol=4,frameon=False,fontsize=12,title='Change-map colors (actual pixels; no dilation)',title_fontsize=11)
    fig.savefig(HERE/'figures/pipeline.png',dpi=180,facecolor='white');plt.close(fig)
    counts=np.array(json.loads((HERE/'tables.json').read_text())['counts'],dtype=np.int64)
    total=int(counts.sum());same=int(np.trace(counts));bio=counts[1:,1:];cross_bio=int(bio.sum()-np.trace(bio));bg_bio=int(counts[0,1:].sum()+counts[1:,0].sum())
    assert same+cross_bio+bg_bio==total
    evidence=dict(source='Frozen 200-patch training GT adjacency counts, before smoothing; both orientations counted so proportions equal undirected-edge proportions',counts_sha256=hashlib.sha256((HERE/'tables.json').read_bytes()).hexdigest(),oriented_counts=dict(total=total,same_label=same,different_biological_labels=cross_bio,background_biological=bg_bio),percent=dict(same_label=100*same/total,different_biological_labels=100*cross_bio/total,background_biological=100*bg_bio/total),scope='Descriptive training-GT graph statistic; not test-prediction contacts or a causal intervention. No experiment rerun.',examples=examples)
    (HERE/'deliverable_diagnostics.json').write_text(json.dumps(evidence,indent=2)+'\n')

if __name__=='__main__':main()

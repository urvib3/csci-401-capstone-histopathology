"""Scientific report figures from saved results; never reruns/tunes predictions."""
import json,os,sys,time
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
HERE=Path(__file__).resolve().parent
COLORS=['#202733','#d94747','#45a5e6','#64b967','#c993dc','#e8bf46']
SHORT=['BG','Neo','Immune','Stroma','Dead','Non-Neo']

def main():
    r=json.loads((HERE/'results.json').read_text());tables=json.loads((HERE/'tables.json').read_text());out=HERE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    c=np.array(tables['final']);fig,ax=plt.subplots(figsize=(6,5.2),layout='constrained');im=ax.imshow(c,cmap='YlOrRd',vmin=0)
    ax.set_xticks(range(6),SHORT,rotation=25,ha='right');ax.set_yticks(range(6),SHORT)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',color='white' if c[i,j]>c.max()*.6 else '#202733',fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.8,label='Compatibility cost (mean off-diagonal = 1)');ax.set_title('Frozen learned + rules table',fontsize=17,pad=15);fig.savefig(out/'final_table.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.3),layout='constrained');names=['Wrong → correct','Correct → wrong','Wrong → other wrong'];values=[r['wrong_to_correct'],r['correct_to_wrong'],r['wrong_to_different_wrong']]
    bars=ax.barh(names,values,color=['#278657','#c24d4d','#9b91ab']);ax.invert_yaxis();ax.bar_label(bars,fmt='{:,.0f}',padding=5);ax.set_xlim(0,max(values)*1.23);ax.set_xlabel('Changed pixels');fig.savefig(out/'fixes_breaks.png',dpi=180);plt.close(fig)
    delta=lambda x:x['wcsp']['biological_miou']-x['baseline']['biological_miou']
    good=sorted([x for x in r['patches'] if delta(x)>0],key=delta,reverse=True)
    bad=sorted([x for x in r['patches'] if delta(x)<0],key=delta)
    selected=[]
    for kind,group in [('improvement',good),('degradation',bad)]:
        if not group:continue
        row=group[0];i=row['image_index'];selected.append(dict(kind=kind,image_index=i,tissue=row['tissue'],delta=delta(row),fixes=row['wrong_to_correct'],breaks=row['correct_to_wrong']))
        z=np.load(HERE/f'predictions/patch_{i:04d}.npz');rgb=np.load(HERE/f'cache/rgb_{i:04d}.npy')[46:210,46:210]
        fig,axes=plt.subplots(1,4,figsize=(12,3.9),layout='constrained')
        for ax,img,title in zip(axes,[rgb,z['gt'],z['baseline'],z['wcsp']],['Histology','GT','Raw TP argmax','WCSP β = 0.30']):
            ax.imshow(img,cmap=None if img.ndim==3 else ListedColormap(COLORS),vmin=None if img.ndim==3 else 0,vmax=None if img.ndim==3 else 5,interpolation='nearest');ax.set_title(title,fontsize=14);ax.axis('off')
        fig.suptitle(f"{kind.title()}: Fold2/{i:04d} • {row['tissue']} • Δ biological mIoU {delta(row):+.4f} • fixes {row['wrong_to_correct']:,}, breaks {row['correct_to_wrong']:,}",fontsize=13)
        fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(COLORS,SHORT)],loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.02));fig.savefig(out/f'{kind}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (HERE/'selected_examples.json').write_text(json.dumps(dict(policy='Largest positive and largest negative fixed-five-class patch mIoU delta; illustrative extremes, not representative sampling.',examples=selected),indent=2)+'\n')
    for label,key in [('baseline','baseline_confusion'),('wcsp','wcsp_confusion'),('transition','transition')]:
        np.savetxt(HERE/f'{label}_matrix.csv',np.array(r[key]),delimiter=',',fmt='%d',header=','.join(SHORT),comments='')
    with (HERE/'per_class.csv').open('w') as f:
        f.write('class,baseline_iou,wcsp_iou\n')
        for name,a,b in zip(SHORT,r['baseline']['iou'],r['wcsp']['iou']):f.write(f'{name},{a:.9f},{b:.9f}\n')

if __name__=='__main__':
    if '--wait' in sys.argv:
        deadline=time.monotonic()+1800
        while not (HERE/'results.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Frozen evaluation did not finish in 30 minutes')
            time.sleep(2)
    main()

"""Independently recount saved predictions, then plot development sweep."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--data-root',type=Path)
args=parser.parse_args()
HERE=args.results.resolve()
r=json.loads((HERE/'results.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
agg=r['aggregate'];betas=[x['beta'] for x in agg];chosen=r['selected_beta'];files=sorted((HERE/'predictions').glob('*.npz'))
assert len(files)==20
for row in agg:
 cm=np.zeros((6,6),int);tot=np.zeros(4,int)
 for f in files:
  z=np.load(f);g=z['gt'];a=z['baseline'];b=z[f'beta_{row["beta"]:g}']
  cm+=np.bincount((g.astype(int)*6+b).ravel(),minlength=36).reshape(6,6)
  tot+=np.array([(a!=b).sum(),((a!=g)&(b==g)).sum(),((a==g)&(b!=g)).sum(),((a!=b)&(a!=g)&(b!=g)).sum()])
  if row['beta']==0:assert np.array_equal(a,b)
 union=cm.sum(0)+cm.sum(1)-np.diag(cm);iou=np.divide(np.diag(cm),union,out=np.zeros(6,float),where=union>0)
 assert np.allclose(iou,row['iou'],atol=1e-14,rtol=0)
 assert abs(iou[1:].mean()-row['biological_miou'])<1e-14
 assert abs(np.trace(cm)/cm.sum()-row['pixel_accuracy'])<1e-14
 assert tot.tolist()==[row[k] for k in ['changed_pixels','wrong_to_correct','correct_to_wrong','wrong_to_different_wrong']]
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].plot(betas,[100*x['biological_miou'] for x in agg],'o-',label='Biological mIoU')
ax[0].axhline(100*agg[0]['biological_miou'],color='gray',ls=':',label='Raw TP baseline')
ax[0].axvline(chosen,color='green',ls='--',label=f'Dev selected beta = {chosen:g}')
ax[0].set(xlabel='Beta',ylabel='Pooled biological mIoU (%)',title='Frozen learned table: development selection');ax[0].legend(fontsize=8)
ax[1].plot(betas,[x['wrong_to_correct'] for x in agg],'o-',label='Fixes');ax[1].plot(betas,[x['correct_to_wrong'] for x in agg],'o-',label='Breaks')
ax[1].set(xlabel='Beta',ylabel='Pixels',title='Corrections versus damage');ax[1].legend()
fig.savefig(HERE/'beta_sweep.png',dpi=160);plt.close(fig)
best=next(x for x in agg if x['beta']==chosen);fixed=next(x for x in agg if x['beta']==.3)
names=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic']
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');xx=np.arange(6)
for j,(row,label) in enumerate([(agg[0],'Raw TP'),(fixed,'Beta 0.30'),(best,f'Dev selected {chosen:g}')]):ax.bar(xx+(j-1)*.25,100*np.array(row['iou']),.25,label=label)
ax.set_xticks(xx,names,rotation=15);ax.set_ylabel('Pooled IoU (%)');ax.legend();fig.savefig(HERE/'per_class.png',dpi=160);plt.close(fig)
def score(g,p):
 vals=[]
 for k in range(1,6):
  union=((g==k)|(p==k)).sum();vals.append(((g==k)&(p==k)).sum()/union if union else 0)
 return np.mean(vals)
rank=[]
for f in files:
 z=np.load(f);rank.append((score(z['gt'],z[f'beta_{chosen:g}'])-score(z['gt'],z['baseline']),f))
palette=ListedColormap(['#202833','#d94444','#40a6df','#60b766','#c58cdb','#e7bd40'])
repo=next(p for p in Path(__file__).resolve().parents if (p/'benchmarks/pannuke').is_dir());data=args.data_root or repo/'benchmarks/pannuke/data';rgb=np.load(data/'images/fold2/images.npy',mmap_mode='r')
for title,(delta,f) in [('largest_gain',max(rank)),('largest_loss',min(rank))]:
 z=np.load(f);idx=int(f.stem.split('_')[1]);fig,axes=plt.subplots(1,5,figsize=(14,3),layout='constrained')
 panels=[rgb[idx][46:210,46:210].astype(np.uint8),z['gt'],z['baseline'],z['beta_0.3'],z[f'beta_{chosen:g}']]
 for ax,img,label in zip(axes,panels,['H&E','GT','Raw TP','Beta 0.30',f'Beta {chosen:g}']):
  ax.imshow(img,**({} if img.ndim==3 else dict(cmap=palette,vmin=0,vmax=5)));ax.set_title(label);ax.axis('off')
 fig.suptitle(f'Development patch {idx}; selected-beta patch mIoU change {100*delta:+.3f} pp (illustrative extreme)')
 fig.savefig(HERE/f'{title}.png',dpi=150);plt.close(fig)
text=['# Frozen learned-table beta sweep — development only','',
'The D5 learned 80/20 data/prior table is unchanged. Twenty development patches are disjoint from its 200 fitting patches, its 20 test patches, and recorded previous selections. Selection used GT coverage and tissue diversity before inference. No final test was run.','',
f'Selected beta by pooled fixed-five-class mIoU: **{chosen:g}**. This is a development selection result, not evidence of generalization.','',
'| Beta | Bio mIoU % | Six-class mIoU % | Accuracy % | Changed | Fixes | Breaks | Local minima /20 | Solver seconds |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in agg:text.append(f'| {x["beta"]:g} | {100*x["biological_miou"]:.4f} | {100*x["six_class_miou"]:.4f} | {100*x["pixel_accuracy"]:.4f} | {x["changed_pixels"]} | {x["wrong_to_correct"]} | {x["correct_to_wrong"]} | {x["local_minima"]} | {x["solver_seconds"]:.3f} |')
text+=['',f'Inference once per patch: {r["inference_seconds"]:.3f}s total; model setup {r["model_setup_seconds"]:.3f}s. Solver timing includes energy and local-minimum audits. Do not sum inference times repeated across per-beta CSV rows.','',
'## Interpretation and limits','',
'Beta selection changes only the global multiplier. All runs initialize independently from raw TP argmax. Same copied D5 objective, 24 neighbors, eta=1, strict raster ICM, maximum 100 rounds. Beta zero matches argmax exactly. Every saved prediction was independently recounted to verify all reported IoUs, accuracy and change counts. Local minima do not imply global optima.',
'','Development membership is class-enriched, not prevalence-representative. Historical exclusions cover recorded selections, not all possible unrecorded use. Patient/slide separation and HoVer-Net pretraining exposure are unverified. A new untouched evaluation set is needed after selection; do not tune against the earlier 20-patch result.',
'','Local environment uses TIAToolbox 2.0.1 and PyTorch 2.9.1 rather than D5 recorded 2.1.2/2.12.0. Checkpoint SHA256 is identical, channel labels are checked, raw six-channel TP forward is used. Absolute scores also differ because development membership differs.','',
'## Visuals','',f'![Sweep](beta_sweep.png)',f'![Per-class](per_class.png)',f'![Largest gain](largest_gain.png)',f'![Largest loss](largest_loss.png)']
(HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
(HERE/'verification.json').write_text(json.dumps(dict(saved_prediction_recount=True,beta_zero_exact=True,patches=20,betas=len(betas),selected_beta=chosen),indent=2))
print('Verified all saved predictions and generated report/figures.')

"""Post-hoc descriptive summaries; no model selection or optimization."""
import csv,json
from pathlib import Path
import numpy as np
from core import CLASSES,bootstrap,metrics

HERE=Path(__file__).resolve().parent

def main():
    report={};arrays={}
    for version in [1,2]:
        dest=HERE/f'version_{version}'
        if not (dest/'results.json').exists():continue
        r=json.loads((dest/'results.json').read_text());arr=np.load(dest/'confusions.npz');arrays[version]=arr['wcsp'];tr=np.zeros((6,6),int);fix=np.zeros((6,6),int);brk=np.zeros((6,6),int)
        for row in r['patches']:
            a=np.load(dest/f'patch_{row["image_index"]:04d}.npz');raw=a['raw'];pred=a['pred'];gt=a['gt'];changed=raw!=pred
            tr+=np.bincount((raw.astype(int)*6+pred).ravel(),minlength=36).reshape(6,6)
            for out,mask in [(fix,changed&(pred==gt)),(brk,changed&(raw==gt))]:
                out+=np.bincount((raw[mask].astype(int)*6+pred[mask]).ravel(),minlength=36).reshape(6,6)
        transitions=[]
        for a in range(6):
            for b in range(6):
                if a!=b and tr[a,b]:transitions.append(dict(source=CLASSES[a],target=CLASSES[b],changes=int(tr[a,b]),fixes=int(fix[a,b]),breaks=int(brk[a,b]),different_wrong=int(tr[a,b]-fix[a,b]-brk[a,b])))
        transitions.sort(key=lambda x:x['changes'],reverse=True)
        with (dest/'class_transitions.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['source','target','changes','fixes','breaks','different_wrong']);w.writeheader();w.writerows(transitions)
        with (dest/'per_class.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['class','raw_iou','wcsp_iou','delta'])
            for k,name in enumerate(CLASSES):w.writerow([name,r['raw']['iou'][k],r['wcsp']['iou'][k],r['wcsp']['iou'][k]-r['raw']['iou'][k]])
        levels=[]
        for l in range(6):
            levels.append(dict(level=l+1,changed=sum(x['solver']['levels'][l]['changed'] for x in r['patches']),variables=sum(x['solver']['levels'][l]['variables'] for x in r['patches']),parent_disagreement_before=sum(x['solver']['levels'][l]['parent_disagreement_before'] for x in r['patches']),parent_disagreement_after=sum(x['solver']['levels'][l]['parent_disagreement_after'] for x in r['patches'])))
        report[f'version_{version}']=dict(transitions=transitions,levels=levels)
    if 1 in arrays and 2 in arrays:
        differences=[]
        for record in r['patches']:
            i=record['image_index'];a=np.load(HERE/f'version_1/patch_{i:04d}.npz')['pred'];b=np.load(HERE/f'version_2/patch_{i:04d}.npz')['pred'];differences.append(int((a!=b).sum()))
        report['version_2_minus_version_1']=dict(delta=metrics(arrays[2].sum(0))['biological_miou']-metrics(arrays[1].sum(0))['biological_miou'],ci95=bootstrap(arrays[1],arrays[2]),different_pixels=sum(differences),different_patches=sum(x>0 for x in differences))
    (HERE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k.startswith('version_2_minus')},indent=2))

if __name__=='__main__':main()

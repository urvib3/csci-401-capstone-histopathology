"""Paired scaling follow-up, then original-V1 sibling ablation; no D6 edits."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from core import *

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'hierarchy_20261109'
ROOT=HERE.parents[2]
def dump(p,x):
    p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def inputs(i):
    return np.load(PREV/f'cache/prob_{i:04d}.npz')['p'].astype(float),np.load(PREV/f'cache/gt_{i:04d}.npy')

def run_condition(name,tables,g,params,split):
    dest=HERE/name;dest.mkdir(exist_ok=True);assert not (dest/'results.json').exists()
    rows=[];aa=[];bb=[];transitions=np.zeros((6,6),int)
    for record in split['test']:
        i=record['image_index'];p,gt=inputs(i);raw=p.argmax(-1).astype(np.uint8)
        y,info=solve(p,tables,g,**params);pred=y[:65536].reshape(256,256)
        if name=='biological':
            original=np.load(PREV/f'version_1/patch_{i:04d}.npz');np.testing.assert_array_equal(pred,original['pred'])
        a=confusion(gt,raw);b=confusion(gt,pred);aa.append(a);bb.append(b)
        fix=(raw!=gt)&(pred==gt);brk=(raw==gt)&(pred!=gt);changed=raw!=pred
        boundary=np.zeros(gt.shape,bool)
        boundary[:-1]|=gt[:-1]!=gt[1:];boundary[1:]|=gt[:-1]!=gt[1:];boundary[:,:-1]|=gt[:,:-1]!=gt[:,1:];boundary[:,1:]|=gt[:,:-1]!=gt[:,1:]
        from scipy.ndimage import binary_dilation
        near=binary_dilation(boundary,iterations=2);sp=np.sort(p,-1);margin=sp[...,-1]-sp[...,-2]
        confidence={key:dict(n=int(mask.sum()),margin_sum=float(margin[mask].sum())) for key,mask in [('fixed',fix),('broken',brk),('unchanged_correct',(raw==gt)&~changed)]}
        row=dict(**record,raw=metrics(a),wcsp=metrics(b),changed=int(changed.sum()),fixes=int(fix.sum()),breaks=int(brk.sum()),different_wrong=int((changed&~fix&~brk).sum()),net=int(fix.sum()-brk.sum()),boundary=dict(fixes=int((fix&near).sum()),breaks=int((brk&near).sum())),interior=dict(fixes=int((fix&~near).sum()),breaks=int((brk&~near).sum())),confidence=confidence,solver=info)
        rows.append(row);tr=confusion(raw,pred);transitions+=tr
        np.savez_compressed(dest/f'patch_{i:04d}.npz',pred=pred,levels=y,transition=tr)
        print(name,i,'delta',round(row['wcsp']['biological_miou']-row['raw']['biological_miou'],6),flush=True)
    aa=np.array(aa);bb=np.array(bb);ds=np.array([r['wcsp']['biological_miou']-r['raw']['biological_miou'] for r in rows])
    rawm=metrics(aa.sum(0));wm=metrics(bb.sum(0))
    r=dict(name=name,params=params,raw=rawm,wcsp=wm,delta=wm['biological_miou']-rawm['biological_miou'],ci95_vs_raw=bootstrap(aa,bb),wins=int((ds>1e-12).sum()),losses=int((ds< -1e-12).sum()),ties=int((abs(ds)<=1e-12).sum()),patches=rows,changed=sum(x['changed'] for x in rows),fixes=sum(x['fixes'] for x in rows),breaks=sum(x['breaks'] for x in rows),different_wrong=sum(x['different_wrong'] for x in rows),seconds=sum(x['solver']['seconds'] for x in rows),all_local_minima=all(x['solver']['verified_local_minimum'] for x in rows),transitions=transitions)
    dump(dest/'results.json',r);np.savez_compressed(dest/'confusions.npz',raw=aa,wcsp=bb)
    with (dest/'per_patch.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['image_index','tissue','raw_bio','wcsp_bio','delta','fixes','breaks','net','changed','seconds'])
        for x in rows:w.writerow([x['image_index'],x['tissue'],x['raw']['biological_miou'],x['wcsp']['biological_miou'],x['wcsp']['biological_miou']-x['raw']['biological_miou'],x['fixes'],x['breaks'],x['net'],x['changed'],x['solver']['seconds']])
    print('RESULT',name,{k:r[k] for k in ['delta','wins','losses','ties','fixes','breaks','seconds']},flush=True)
    return r

def main():
    assert not (HERE/'protocol.json').exists(),'Preserve completed experiment; use a new folder.'
    s=json.loads((PREV/'split.json').read_text());frozen=json.loads((PREV/'frozen_config.json').read_text());tabs=json.loads((PREV/'tables.json').read_text());bio=np.array(tabs['final']);potts=np.stack([1-np.eye(6)]*6);g=graph()
    before={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'deliverables/deliverable_6.md',PREV/'core.py',PREV/'run.py',PREV/'frozen_config.json',PREV/'version_1/results.json',PREV/'version_2/results.json']}
    dump(HERE/'protocol.json',dict(seed=20261109,source_split_sha256=sha(PREV/'split.json'),source_tables_sha256=sha(PREV/'tables.json'),source_hashes={n:sha(HERE/n) for n in ['core.py','run.py']},preserve_before=before,scaling='Block unaries x area; sibling costs x side; parent-child mismatch x child area. Keep six levels, four-neighbor graph, and learned tables; this adapts BCSS scaling rather than copying its complete model.',sibling_ablation='Original unscaled V1; gamma=0; sibling=.3 and parent=.1; only same-level cost tables change.',reuse='Exact original 100 patches/probabilities/GT, 12 development patches; exploratory follow-up; four test IDs overlap newly fetched beta development sweep.',scaling_grid_parent=[.01,.03,.1,.3,.5],gamma=.3,sibling=.3,stopping='Bidirectional strict-decrease ICM; max100; independent local audit',bootstrap='5000 paired patch resamples; fixed seed20261109; fixed-five-class pooled mIoU; zero-union=0',d6='Deferred by direct user instruction; no edits now.'))
    dump(HERE/'sibling_tables.json',dict(biological=bio,potts=potts,parent_only=np.zeros_like(bio)))
    # Predeclare, then choose scaling strength on original development patches only.
    grid=[]
    for parent in [.01,.03,.1,.3,.5]:
        cm=np.zeros((6,6),np.int64)
        for rec in s['dev']:
            p,gt=inputs(rec['image_index']);y,_=solve(p,bio,g,.3,parent,.3,scaled=True);cm+=confusion(gt,y[:65536].reshape(256,256))
        grid.append(dict(parent=parent,score=metrics(cm)['biological_miou']));print('SCALING DEV',grid[-1],flush=True)
    best=max(grid,key=lambda x:x['score'])['parent'];dump(HERE/'scaling_frozen.json',dict(sibling=.3,gamma=.3,selected_parent=best,development=grid,selected_before_comparison=True))
    results={}
    for name,params in [('scaled_matched',dict(sibling=.3,parent=.1,gamma=.3,scaled=True)),('scaled_calibrated',dict(sibling=.3,parent=best,gamma=.3,scaled=True))]:
        results[name]=run_condition(name,bio,g,params,s)
    print('SCALING COMPLETE; starting original-V1 sibling ablation.',flush=True)
    for name,tables in [('parent_only',np.zeros_like(bio)),('potts',potts),('biological',bio)]:
        results[name]=run_condition(name,tables,g,dict(sibling=.3,parent=.1,gamma=0,scaled=False),s)
    arrays={k:np.load(HERE/k/'confusions.npz')['wcsp'] for k in results};raw=np.load(HERE/'parent_only/confusions.npz')['raw']
    old2=np.load(PREV/'version_2/confusions.npz')['wcsp'];arrays['unscaled_block_unaries']=old2
    pairs={}
    for a,b in [('scaled_matched','unscaled_block_unaries'),('scaled_calibrated','unscaled_block_unaries'),('parent_only','raw'),('potts','parent_only'),('biological','parent_only'),('biological','potts')]:
        x=raw if b=='raw' else arrays[b];y=arrays[a];d=metrics(y.sum(0))['biological_miou']-metrics(x.sum(0))['biological_miou'];pairs[a+' minus '+b]=dict(delta=d,ci95=bootstrap(x,y),per_class=np.array(metrics(y.sum(0))['iou'])-np.array(metrics(x.sum(0))['iou']))
    dump(HERE/'comparisons.json',pairs)
    for name,digest in before.items():assert sha(ROOT/name)==digest,'Preserved file changed: '+name
    dump(HERE/'preservation.json',dict(previous_experiment_unchanged=True,deliverable_6_unchanged=True,reference_v1_reproduced_exactly=True))
    print('COMPARISONS',json.dumps(pairs,default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':main()

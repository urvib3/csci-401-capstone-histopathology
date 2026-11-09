"""Frozen full-patch hierarchical experiment; no test-outcome parameter selection."""
import argparse,csv,hashlib,importlib.util,json,platform,re,sys,time,zipfile
from pathlib import Path
import numpy as np
from core import *

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'pannuke_d5'
LEGACY=ROOT.parent/'cs401-pannuke/pannuke_d5'
SEED=20261109

def dump(p,x):
    p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def old_module():
    spec=importlib.util.spec_from_file_location('prior_run',OLD/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def exclusions():
    out={}
    def add(i,source):
        if isinstance(i,int) and 0<=i<2523:out.setdefault(str(i),set()).add(source)
    prior=json.loads((OLD/'previous_use_exclusions.json').read_text())
    for i,src in prior.items():out[i]=set(src)
    def walk(x,src,key=''):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ['image_index','patch_index','image_idx']:add(v,src)
                walk(v,src,k)
        elif isinstance(x,list):
            if 'indices' in key or key in ['train_ids','test_ids','patch_ids','development_ids']:
                for i in x:add(i,src)
            for v in x:walk(v,src)
    for base,label in [(OLD,'current_d5'),(LEGACY,'legacy_d5')]:
        for p in base.rglob('*'):
            if not p.is_file() or 'cache' in p.parts:continue
            rel=label+'/'+str(p.relative_to(base))
            if p.suffix in ['.npy','.npz','.json']:
                for m in re.finditer(r'(?:image|patch)[_-]0*(\d+)',str(p.relative_to(base))):add(int(m[1]),rel)
            if p.suffix=='.json' and any(x in p.name for x in ['config','split','selection','summary','fit']) and p.stat().st_size<5_000_000:
                try:walk(json.loads(p.read_text()),rel)
                except ValueError:pass
            elif p.suffix=='.csv' and p.name not in ['pannuke_patch_catalog.csv','pannuke_candidate_patches.csv']:
                try:
                    for row in csv.DictReader(p.open()):
                        for k in ['image_index','patch_index','image_idx']:
                            if row.get(k,'').isdigit():add(int(row[k]),rel)
                except (ValueError,csv.Error):pass
    return {i:sorted(v) for i,v in out.items()}

def prepare():
    assert not (HERE/'split.json').exists()
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    m=old_module();s=json.loads((OLD/'split.json').read_text());train=[r['image_index'] for r in s['train']]
    ex=exclusions()
    for i in train:ex.setdefault(str(i),[]).append('existing_200_table_fit')
    z,n=m.members('types.npy')
    with z,z.open(n) as f:tissue=np.load(f)
    gt=np.load(OLD/'cache/all_semantic.npy',mmap_mode='r')
    present=np.array([(np.bincount(x.ravel(),minlength=6)>0) for x in gt])
    rng=np.random.default_rng(SEED);eligible=rng.permutation([i for i in range(len(gt)) if str(i) not in ex]).tolist()
    def choose(pool,n,target):
        result=[];cov=np.zeros(6,int);ts=set()
        for _ in range(n):
            best=max(pool,key=lambda i:(int((present[i,1:]&(cov[1:]<target)).sum()),int(str(tissue[i]) not in ts)))
            result.append(best);pool.remove(best);cov+=present[best];ts.add(str(tissue[best]))
        return result
    # Select independent comparison set first; use GT coverage, never predictions.
    test=choose(eligible,100,8);dev=choose(eligible,12,2)
    assert not set(dev)&set(test) and not (set(dev+test)&set(train))
    records=lambda ids:[dict(image_index=i,tissue=str(tissue[i])) for i in ids]
    split=dict(seed=SEED,train=records(train),dev=records(dev),test=records(test),selection='Seeded GT-coverage and tissue-diversity selection using existing central GT catalog; full-patch coverage reported after extraction.',training='Same 200 fitting patches as previous D5; not new independent training data.',independence='Patch-disjoint from identifiable previous uses; no patient/slide identifiers; checkpoint exposure unknown.')
    dump(HERE/'split.json',split);dump(HERE/'exclusions.json',ex)
    full={};needed=set(train+dev+test)
    print('Extracting full masks for',len(needed),'patches',flush=True)
    for i,mask in m.stream_arrays('masks.npy',needed):
        y=np.zeros((256,256),np.uint8)
        for k in range(5):y[mask[...,k]>0]=k+1
        full[i]=y;np.save(cache/f'gt_{i:04d}.npy',y)
    tables=[];evidence=[]
    for size in SIZES:
        n=sum((counts(majority(full[i],size)) for i in train),start=np.zeros((6,6),np.int64));tab,ev=fit_table(n);tables.append(tab);evidence.append(ev)
    dump(HERE/'tables.json',dict(sizes=SIZES,final=tables,evidence=evidence,prior=PRIOR,blend=[.8,.2],background='learned independently at each scale',ties='Lowest class ID wins tied GT block plurality; Background ID 0.'))
    for l,tab in enumerate(tables):np.savetxt(HERE/f'table_level_{l+1}.csv',tab,delimiter=',')
    coverage={}
    for name,ids in [('train',train),('dev',dev),('test',test)]:
        c=np.array([np.bincount(full[i].ravel(),minlength=6) for i in ids]);coverage[name]=dict(patches=len(ids),class_pixels=c.sum(0),patch_coverage=(c>0).sum(0),tissues=dict(zip(*np.unique([str(tissue[i]) for i in ids],return_counts=True))))
        with (HERE/f'{name}_manifest.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['image_index','tissue']+CLASSES)
            for i,row in zip(ids,c):w.writerow([i,str(tissue[i])]+row.tolist())
    dump(HERE/'coverage.json',coverage)
    print('Extracting full RGB',flush=True)
    for i,img in m.stream_arrays('images.npy',dev+test):
        assert np.array_equal(img,img.astype(np.uint8));np.save(cache/f'rgb_{i:04d}.npy',img.astype(np.uint8))
    dump(HERE/'prepared.json',dict(complete=True))
    print('Prepared',json.dumps(coverage,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x)),flush=True)

def infer():
    import torch
    m=old_module();start=time.perf_counter();model,hashes=m.load_model()
    assert m.sha(m.DATA/'cache/models/hovernet_fast-pannuke.pth')==json.loads((OLD/'frozen_config.json').read_text())['checkpoint']['sha256']
    while not (HERE/'prepared.json').exists():time.sleep(2)
    split=json.loads((HERE/'split.json').read_text());rows=[]
    print('Model loaded',time.perf_counter()-start,flush=True)
    for record in split['dev']+split['test']:
        i=record['image_index'];path=HERE/f'cache/prob_{i:04d}.npz'
        if path.exists():continue
        image=np.load(HERE/f'cache/rgb_{i:04d}.npy');padded=np.pad(image,((46,46),(46,46),(0,0)),mode='reflect')
        total=np.zeros((256,256,6),float);weight=np.zeros((256,256,1),float);t=time.perf_counter()
        for row in [0,92]:
            for col in [0,92]:
                tile=np.ascontiguousarray(padded[row:row+256,col:col+256]);x=torch.from_numpy(tile.astype(np.float32)).permute(2,0,1)[None]
                with torch.inference_mode():z=model(x)['tp'][0].permute(1,2,0).cpu().numpy().astype(float)
                assert z.shape==(164,164,6);z-=z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
                total[row:row+164,col:col+164]+=p;weight[row:row+164,col:col+164]+=1
        assert weight.min()>0;p=total/weight
        assert np.isfinite(p).all() and np.allclose(p.sum(-1),1)
        dt=time.perf_counter()-t;np.savez_compressed(path,p=p.astype(np.float32),seconds=dt)
        print('infer',i,round(dt,3),flush=True);rows.append(dict(image_index=i,seconds=dt))
    dump(HERE/'inference.json',dict(checkpoint_sha256=m.sha(m.DATA/'cache/models/hovernet_fast-pannuke.pth'),architecture_sha256=hashes,model_setup_seconds=time.perf_counter()-start-sum(r['seconds'] for r in rows),tiles=[[0,0],[0,92],[92,0],[92,92]],input_padding=46,padding='reflect; synthetic border context, not original surrounding slide',fusion='uniform mean of overlapping six-channel TP softmax probabilities',crop='none; all original 256x256 pixels',numpy=np.__version__,torch=torch.__version__,python=platform.python_version(),records=rows))

def load_inputs(i):
    return np.load(HERE/f'cache/prob_{i:04d}.npz')['p'].astype(float),np.load(HERE/f'cache/gt_{i:04d}.npy')

def calibrate():
    assert not (HERE/'frozen_config.json').exists()
    split=json.loads((HERE/'split.json').read_text());tables=np.array(json.loads((HERE/'tables.json').read_text())['final']);g=graph();rows=[]
    # Small preregistered shared-strength grid, development data only.
    for sibling,parent in [(.1,.03),(.3,.1),(.6,.3)]:
        cms=[]
        for r in split['dev']:
            p,gt=load_inputs(r['image_index']);y,info=solve(p,tables,g,sibling,parent);cms.append(confusion(gt,y[:65536].reshape(256,256)))
        score=metrics(sum(cms))['biological_miou'];rows.append(dict(sibling=sibling,parent=parent,score=score));print('dev V1',rows[-1],flush=True)
    best=max(rows,key=lambda x:x['score']);cfg=dict(seed=SEED,sibling=best['sibling'],parent=best['parent'],block_gamma_candidates=[.1,.3,1.0],max_rounds=100,graph=g['counts'],shared_selection=rows,objective='sum pixel U + gamma sum block U + sibling/4 sum scale-table costs + parent sum label-mismatch',block_meaning='soft dominant class; no exact majority requirement',sibling_graph='4-connected spatial adjacency at every level, including adjacent blocks with different immediate parents',prior='Previously user-specified revised D5 biological preference matrix retained; separately learned and normalized six-class table per scale; .8 data/.2 prior.',solver='bidirectional deterministic ICM; strict decreases, local optimum only',init='identical in both versions: raw pixel argmax and averaged-probability block argmax',evaluation='100 fixed patches; pooled five-biological-class mIoU; six-class and per-class metrics also reported',trigger='Run version 2 if V1 pooled biological mIoU does not exceed raw, or any patch biological mIoU worsens.',split_sha256=sha(HERE/'split.json'),tables_sha256=sha(HERE/'tables.json'),source_sha256={n:sha(HERE/n) for n in ['core.py','run.py']})
    dump(HERE/'frozen_config.json',cfg)

def run_version(version):
    cfg=json.loads((HERE/'frozen_config.json').read_text());split=json.loads((HERE/'split.json').read_text());tables=np.array(json.loads((HERE/'tables.json').read_text())['final']);g=graph()
    assert sha(HERE/'tables.json')==cfg['tables_sha256'] and sha(HERE/'split.json')==cfg['split_sha256']
    dest=HERE/f'version_{version}';assert not (dest/'results.json').exists();dest.mkdir(exist_ok=True)
    gamma=0
    if version==2:
        dev=[]
        for gamma in cfg['block_gamma_candidates']:
            cm=np.zeros((6,6),np.int64)
            for r in split['dev']:
                p,gt=load_inputs(r['image_index']);y,inf=solve(p,tables,g,cfg['sibling'],cfg['parent'],gamma);cm+=confusion(gt,y[:65536].reshape(256,256))
            dev.append(dict(gamma=gamma,score=metrics(cm)['biological_miou']));print('dev V2',dev[-1],flush=True)
        gamma=max(dev,key=lambda x:x['score'])['gamma'];dump(dest/'frozen_block_config.json',dict(gamma=gamma,selection=dev,selection_data='12 development patches only; shared weights unchanged'))
    rows=[];raws=[];cms=[];alllevels=[]
    for record in split['test']:
        i=record['image_index'];p,gt=load_inputs(i);raw=p.argmax(-1).astype(np.uint8)
        y,info=solve(p,tables,g,cfg['sibling'],cfg['parent'],gamma);pred=y[:65536].reshape(256,256);a=confusion(gt,raw);b=confusion(gt,pred);raws.append(a);cms.append(b)
        changed=raw!=pred;fix=(raw!=gt)&(pred==gt);brk=(raw==gt)&(pred!=gt)
        boundary=np.zeros(gt.shape,bool);boundary[:-1]|=gt[:-1]!=gt[1:];boundary[1:]|=gt[:-1]!=gt[1:];boundary[:,:-1]|=gt[:,:-1]!=gt[:,1:];boundary[:,1:]|=gt[:,:-1]!=gt[:,1:]
        from scipy.ndimage import binary_dilation
        near=binary_dilation(boundary,iterations=2)
        sortedp=np.sort(p,-1);margin=sortedp[...,-1]-sortedp[...,-2]
        conf={name:dict(pixels=int(mask.sum()),margin_sum=float(margin[mask].sum()),max_probability_sum=float(p.max(-1)[mask].sum())) for name,mask in [('fixed',fix),('broken',brk),('unchanged_correct',(raw==gt)&~changed)]}
        row=dict(**record,raw=metrics(a),wcsp=metrics(b),changed=int(changed.sum()),fixes=int(fix.sum()),breaks=int(brk.sum()),different_wrong=int((changed&~fix&~brk).sum()),net=int(fix.sum()-brk.sum()),boundary=dict(fixes=int((fix&near).sum()),breaks=int((brk&near).sum())),interior=dict(fixes=int((fix&~near).sum()),breaks=int((brk&~near).sum())),confidence=conf,solver=info)
        rows.append(row);alllevels.append(y);np.savez_compressed(dest/f'patch_{i:04d}.npz',gt=gt,raw=raw,pred=pred,levels=y,transition=confusion(raw,pred));dump(dest/f'patch_{i:04d}.json',row)
        print('V',version,i,'delta',round(row['wcsp']['biological_miou']-row['raw']['biological_miou'],6),'fix/break',row['fixes'],row['breaks'],'rounds',info['rounds'],flush=True)
    aa=np.array(raws);bb=np.array(cms);rawm=metrics(aa.sum(0));wm=metrics(bb.sum(0));ds=np.array([r['wcsp']['biological_miou']-r['raw']['biological_miou'] for r in rows])
    result=dict(version=version,gamma=gamma,raw=rawm,wcsp=wm,delta=wm['biological_miou']-rawm['biological_miou'],ci95=bootstrap(aa,bb),wins=int((ds>1e-12).sum()),losses=int((ds< -1e-12).sum()),ties=int((abs(ds)<=1e-12).sum()),patches=rows,fixes=sum(r['fixes'] for r in rows),breaks=sum(r['breaks'] for r in rows),changed=sum(r['changed'] for r in rows),different_wrong=sum(r['different_wrong'] for r in rows),seconds=sum(r['solver']['seconds'] for r in rows),all_local_minima=all(r['solver']['verified_local_minimum'] for r in rows))
    dump(dest/'results.json',result);np.savez_compressed(dest/'confusions.npz',raw=aa,wcsp=bb)
    with (dest/'per_patch.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['image_index','tissue','raw_bio_miou','wcsp_bio_miou','delta','fixes','breaks','changed','rounds','seconds'])
        for r in rows:w.writerow([r['image_index'],r['tissue'],r['raw']['biological_miou'],r['wcsp']['biological_miou'],r['wcsp']['biological_miou']-r['raw']['biological_miou'],r['fixes'],r['breaks'],r['changed'],r['solver']['rounds'],r['solver']['seconds']])
    print('RESULT',version,json.dumps({k:result[k] for k in ['raw','wcsp','delta','ci95','wins','losses','ties','fixes','breaks','seconds']}),flush=True)
    return result

def evaluate():
    r=run_version(1)
    if r['delta']<=0 or r['losses']>0:
        print('V1 non-degradation target not met; starting V2 immediately.',flush=True);run_version(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','calibrate','evaluate']);a=p.parse_args();globals()[a.stage]()

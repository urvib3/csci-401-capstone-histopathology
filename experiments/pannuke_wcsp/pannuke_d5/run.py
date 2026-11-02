"""New standalone experiment: prepare -> fit/freeze -> evaluate, exactly one beta."""
import argparse, ast, csv, hashlib, importlib.metadata, json, math, platform, re, sys, time, warnings, zipfile
from collections import OrderedDict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'benchmarks/pannuke/data'
SEED=20261101
CLASSES=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic Epithelial']

def dump(path,value):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert)+'\n')

def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def members(suffix):
    z=zipfile.ZipFile(DATA/'fold_2.zip'); names=[n for n in z.namelist() if n.endswith('/'+suffix)]
    assert len(names)==1
    return z,names[0]

def stream_arrays(name,indices=None):
    z,n=members(name)
    with z,z.open(n) as f:
        ver=np.lib.format.read_magic(f)
        shape,fort,dtype=(np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f))
        assert not fort and not dtype.hasobject
        size=int(np.prod(shape[1:]))*dtype.itemsize
        chosen=set(range(shape[0])) if indices is None else set(indices)
        # Read forward once; do not extract 12 GB of float64 arrays.
        for i in range(max(chosen)+1):
            raw=f.read(size)
            assert len(raw)==size
            if i in chosen: yield i,np.frombuffer(raw,dtype=dtype).reshape(shape[1:])

def mapping():
    import yaml
    p=ROOT/'.venv/lib/python3.12/site-packages/tiatoolbox/data/pretrained_model.yaml'
    meta=yaml.safe_load(p.read_text())['hovernet_fast-pannuke']
    actual=meta['architecture']['kwargs']['nuc_type_dict']
    assert actual==dict(enumerate(CLASSES)),actual
    return dict(source=str(p),sha256=sha(p),entry='hovernet_fast-pannuke',metadata=meta,verified_mapping=actual)

def exclusions():
    found={}; sources=[]
    def add(i,p):
        if 0<=i<2523: found.setdefault(i,set()).add(str(p.relative_to(ROOT)))
    for base in [ROOT/'experiments',ROOT/'deliverables']:
        for p in sorted(base.rglob('*')):
            if HERE in p.parents or p==HERE: continue
            if p.name in {'pannuke_patch_catalog.csv','pannuke_candidate_patches.csv'}: continue
            for match in re.finditer(r'(?:image|patch)[_-]0*(\d+)',str(p.relative_to(ROOT)),re.I): add(int(match[1]),p)
            if p.is_file() and p.suffix in {'.csv','.json','.md','.py'} and p.stat().st_size<2000000:
                text=p.read_text(errors='replace')
                for match in re.finditer(r'(?:image|patch)[_-]0*(\d+)',text,re.I): add(int(match[1]),p)
                if p.suffix=='.csv':
                    try:
                        for row in csv.DictReader(text.splitlines()):
                            for key in ['image_index','patch_index','image_idx']:
                                if key in row and row[key].strip().isdigit(): add(int(row[key]),p)
                    except (csv.Error,AttributeError): pass
                if p.suffix=='.json':
                    try:
                        def walk(x):
                            if isinstance(x,dict):
                                for k,v in x.items():
                                    if k in ['image_index','patch_index','image_idx'] and isinstance(v,int): add(v,p)
                                    walk(v)
                            elif isinstance(x,list):
                                for v in x: walk(v)
                        walk(json.loads(text))
                    except ValueError: pass
    for i in range(10): add(i,ROOT/'benchmarks/pannuke/data/sample_fold2/README.md')
    return {str(k):sorted(v) for k,v in sorted(found.items())}

def prepare():
    from core import semantic
    assert not (HERE/'split.json').exists(),'Preserve existing split; use a fresh copied directory to reproduce.'
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    z,n=members('types.npy')
    with z,z.open(n) as f: tissues=np.load(f,allow_pickle=False)
    excluded=exclusions(); dump(HERE/'previous_use_exclusions.json',excluded)
    counts=np.zeros((len(tissues),6),np.int64);overlaps=[]
    gts=np.lib.format.open_memmap(cache/'all_semantic.npy',mode='w+',dtype='uint8',shape=(len(tissues),164,164))
    print('Scanning GT class metadata only; no predictions loaded.',flush=True)
    for i,m in stream_arrays('masks.npy'):
        gt,ov=semantic(m);gts[i]=gt;counts[i]=np.bincount(gt.ravel(),minlength=6);overlaps.append(ov)
    gts.flush()
    rng=np.random.default_rng(SEED)
    eligible=np.array([i for i in range(len(tissues)) if str(i) not in excluded])
    order=rng.permutation(eligible).tolist();test=[]
    # Coverage-first greedy policy fixed before inference: aim >=3 patches/class,
    # then maximize tissue diversity; use seeded order for all ties.
    coverage=np.zeros(6,int)
    while len(test)<20:
        best=max((i for i in order if i not in test),key=lambda i:(int(((counts[i,1:]>0)&(coverage[1:]<3)).sum()),int(tissues[i] not in {tissues[j] for j in test})))
        test.append(best);coverage+=(counts[best]>0)
    train=rng.permutation([i for i in eligible if i not in test])[:200].tolist()
    assert len(train)==200 and len(test)==20 and not set(train)&set(test)
    assert (counts[test,1:].sum(0)>0).all()
    def records(ids):
        return [dict(fold=2,image_index=int(i),tissue=str(tissues[i]),class_pixels=counts[i].tolist(),overlap_pixels=overlaps[i]) for i in ids]
    split=dict(seed=SEED,classes=CLASSES,train=records(train),test=records(test),selection='GT coverage >=3 per biological class when feasible, tissue diversity, seeded tie order; train seeded random remainder',
        previous_use_exclusion='Identifiable experiment/development/evaluation/debug IDs and sample first ten excluded. Full-fold inference-only caches are not development selection. Unrecorded prior use cannot be verified.',
        slide_patient_leakage='No slide/patient identifiers in available archive: images, masks, tissue types only. Cannot verify slide/patient separation.',
        held_out='Held out from WCSP table fitting only; pretrained HoVer-Net exposure to Fold 2 unverified.')
    dump(HERE/'split.json',split)
    for name,ids in [('train',train),('test',test)]:
        with (HERE/f'{name}_manifest.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['fold','image_index','tissue']+CLASSES)
            for i in ids:w.writerow([2,i,str(tissues[i])]+counts[i].tolist())
    summary={name:dict(patches=len(ids),class_pixels=counts[ids].sum(0),patch_coverage=(counts[ids]>0).sum(0),tissues=dict(zip(*np.unique(tissues[ids],return_counts=True)))) for name,ids in [('train',train),('test',test)]}
    summary['excluded_patch_count']=len(excluded);summary['eligible_patch_count']=len(eligible)
    dump(HERE/'coverage.json',summary); print(json.dumps(summary,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x),indent=2),flush=True)
    # Save test RGB after freezing membership, without reading predictions.
    for i,image in stream_arrays('images.npy',test):
        assert np.isfinite(image).all() and image.min()>=0 and image.max()<=255
        assert np.array_equal(image,image.astype(np.uint8)), 'RGB cast must be lossless'
        np.save(cache/f'rgb_{i:04d}.npy',image.astype(np.uint8))

def freeze():
    from core import adjacency,fit,HALF
    assert not (HERE/'frozen_config.json').exists()
    split=json.loads((HERE/'split.json').read_text()); gt=np.load(HERE/'cache/all_semantic.npy',mmap_mode='r')
    counts=sum((adjacency(gt[r['image_index']]) for r in split['train']),start=np.zeros((6,6),np.int64))
    tables=fit(counts);dump(HERE/'tables.json',tables)
    for k,v in tables.items():
        if isinstance(v,np.ndarray):np.savetxt(HERE/f'table_{k}.csv',v,delimiter=',',fmt='%.12g')
    config=dict(classes=CLASSES,channel_mapping=mapping(),split_sha256=sha(HERE/'split.json'),
        manifests={n:sha(HERE/n) for n in ['train_manifest.csv','test_manifest.csv']},table=tables['final'],tables_sha256=sha(HERE/'tables.json'),
        seed=SEED,beta=.30,eta=1,smoothing=1,blend_data=.8,blend_prior=.2,
        archive=dict(path=str(DATA/'fold_2.zip'),sha256=sha(DATA/'fold_2.zip')),
        architecture_source_sha256={n:sha(ROOT/'.venv/lib/python3.12/site-packages/tiatoolbox/models/architecture'/n) for n in ['utils.py','hovernet.py']},
        normalization=tables['normalization'],counting='12 undirected offsets; add both orientations, including twice on diagonal; unweighted GT counts',
        equations=dict(J='(N_ab+1)/sum(N+1)',m='sum_b J_ab',raw='max(0,-log(J_ab/(m_a*m_b))); diagonal=0',normalization='divide by mean over all ordered off-diagonal entries',unary='-log(max(P_i(k),1e-12)); subtract per-pixel minimum',pairwise='beta*C[a,b]/Euclidean_distance'),
        crop=[46,210,46,210],neighborhood=dict(half_offsets=HALF,full_neighbors=24,squared_distances=[1,2,4,5,8],undirected_edges_once=True),
        gt_overlap='Highest class ID wins overlapping biological channels; background iff no biological annotation; GT channel 5 ignored as background encoding.',
        solver=dict(name='deterministic raster-order ICM',initialization='raw six-channel TP softmax argmax',acceptance='strict local energy decrease',max_rounds=100,stop='complete round with zero changes',independent_audit_tolerance=1e-10,tie_policy='retain current label unless strictly better; ascending labels among improving ties'),
        metrics=dict(primary='pooled biological mIoU, fixed classes 1..5',secondary='pooled six-class mIoU, fixed classes 0..5',zero_union='IoU=0; never drop classes',bootstrap=dict(replicates=5000,seed=20261102,unit='paired patches',interval='percentile 2.5/97.5')),
        checkpoint=dict(path=str(DATA/'cache/models/hovernet_fast-pannuke.pth'),sha256=sha(DATA/'cache/models/hovernet_fast-pannuke.pth')),
        provenance=dict(git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version,platform=platform.platform(),versions={n:importlib.metadata.version(n) for n in ['numpy','numba','torch','tiatoolbox','matplotlib','PyYAML']},source_hashes={n:sha(HERE/n) for n in ['core.py','run.py']}))
    dump(HERE/'frozen_config.json',config)
    (HERE/'frozen_config.sha256').write_text(sha(HERE/'frozen_config.json')+'\n')
    print('Frozen table/config saved BEFORE test predictions.',flush=True)

def load_model():
    """Compile only installed official inference definitions; avoid WSI imports."""
    import torch
    src=ROOT/'.venv/lib/python3.12/site-packages/tiatoolbox/models/architecture'
    ns=dict(__name__='standalone_hovernet',torch=torch,nn=torch.nn,F=torch.nn.functional,np=np,math=math,warnings=warnings,OrderedDict=OrderedDict,ModelABC=torch.nn.Module)
    hashes={}
    for name,required in [('utils.py',{'centre_crop','centre_crop_to_shape','UpSample2x'}),('hovernet.py',{'TFSamepaddingLayer','DenseBlock','ResidualBlock','HoVerNet'})]:
        p=src/name;tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in required]
        assert {n.name for n in nodes}==required
        for n in nodes:
            if n.name=='HoVerNet': n.body=[m for m in n.body if isinstance(m,ast.FunctionDef) and m.name in {'__init__','forward','_create_decoder_branch'}]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(p),'exec'),ns);hashes[name]=sha(p)
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    model=ns['HoVerNet'](num_types=6,mode='fast')
    model.load_state_dict(torch.load(DATA/'cache/models/hovernet_fast-pannuke.pth',map_location='cpu',weights_only=True),strict=True)
    return model.eval(),hashes

def evaluate():
    from core import probabilities,solve,confusion,metrics,bootstrap
    assert not (HERE/'results.json').exists(),'Results already exist: do not rerun/tune.'
    cfg=json.loads((HERE/'frozen_config.json').read_text());assert sha(HERE/'frozen_config.json')==(HERE/'frozen_config.sha256').read_text().strip()
    assert cfg['split_sha256']==sha(HERE/'split.json') and cfg['tables_sha256']==sha(HERE/'tables.json')
    assert all(sha(HERE/n)==v for n,v in cfg['provenance']['source_hashes'].items())
    assert all(sha(HERE/n)==v for n,v in cfg['manifests'].items())
    assert sha(DATA/'cache/models/hovernet_fast-pannuke.pth')==cfg['checkpoint']['sha256']
    import torch
    t=time.perf_counter();model,hashes=load_model();setup=time.perf_counter()-t
    assert hashes==cfg['architecture_source_sha256']
    split=json.loads((HERE/'split.json').read_text());allgt=np.load(HERE/'cache/all_semantic.npy',mmap_mode='r')
    c=np.array(cfg['table']);out=HERE/'predictions';out.mkdir(exist_ok=True);rows=[];cms=[[],[]]
    transition=np.zeros((6,6),np.int64);triples=np.zeros((6,6,6),np.int64)
    # Compile solver on synthetic data before timing test solves.
    dummy=np.ones((3,3,6))/6;solve(dummy,c)
    for record in split['test']:
        i=record['image_index'];gt=allgt[i];image=np.load(HERE/f'cache/rgb_{i:04d}.npy')
        x=torch.from_numpy(image.astype(np.float32)).permute(2,0,1)[None]
        t=time.perf_counter()
        with torch.inference_mode():raw=model(x)['tp'].detach().cpu().numpy()
        assert raw.shape==(1,6,164,164)
        p=probabilities(raw[0].transpose(1,2,0));infer=time.perf_counter()-t
        baseline=p.argmax(-1).astype(np.uint8)
        zero,zinfo=solve(p,c,beta=0);assert np.array_equal(zero,baseline)
        t=time.perf_counter();refined,info=solve(p,c,beta=.30);runtime=time.perf_counter()-t
        a=confusion(gt,baseline);b=confusion(gt,refined);cms[0].append(a);cms[1].append(b)
        changed=baseline!=refined;fix=int(((baseline!=gt)&(refined==gt)).sum());brk=int(((baseline==gt)&(refined!=gt)).sum());other=int((changed&(baseline!=gt)&(refined!=gt)).sum())
        transition+=confusion(baseline,refined)
        triples+=np.bincount((gt.astype(int)*36+baseline*6+refined).ravel(),minlength=216).reshape(6,6,6)
        row=dict(**record,baseline=metrics(a),wcsp=metrics(b),changed_pixels=int(changed.sum()),wrong_to_correct=fix,correct_to_wrong=brk,wrong_to_different_wrong=other,inference_seconds=infer,solver_seconds=runtime,solver=info,beta_zero_exact=True)
        rows.append(row);np.savez_compressed(out/f'patch_{i:04d}.npz',gt=gt,baseline=baseline,wcsp=refined)
        # Large probabilities/logits are local-only cache, not deliverables.
        np.savez_compressed(HERE/f'cache/tp_{i:04d}.npz',logits=raw[0].transpose(1,2,0),probabilities=p)
        dump(out/f'patch_{i:04d}.json',row)
        print(f'{i:04d}: fixes={fix} breaks={brk} rounds={info["rounds"]} local_min={info["verified_local_minimum"]} inference={infer:.2f}s solve={runtime:.2f}s',flush=True)
    a,b=map(np.array,cms);boot=bootstrap(a,b)
    np.savez_compressed(HERE/'confusions.npz',baseline=a,wcsp=b,transition=transition,gt_baseline_wcsp=triples)
    result=dict(baseline=metrics(a.sum(0)),wcsp=metrics(b.sum(0)),baseline_confusion=a.sum(0),wcsp_confusion=b.sum(0),transition=transition,
        changed_pixels=sum(r['changed_pixels'] for r in rows),wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),wrong_to_different_wrong=sum(r['wrong_to_different_wrong'] for r in rows),
        bootstrap=boot,patches=rows,model_setup_seconds=setup,inference_seconds=sum(r['inference_seconds'] for r in rows),solver_seconds=sum(r['solver_seconds'] for r in rows),
        prediction_provenance=dict(source='fresh CPU model.forward RGB 0..255; internal model normalization; raw tp logits; softmax all six channels; no NP/HV/instances used',architecture_sha256=hashes,checkpoint_sha256=cfg['checkpoint']['sha256'],frozen_config_sha256=sha(HERE/'frozen_config.json')))
    result['biological_miou_difference']=result['wcsp']['biological_miou']-result['baseline']['biological_miou']
    dump(HERE/'results.json',result)
    print(json.dumps({k:result[k] for k in ['baseline','wcsp','biological_miou_difference','wrong_to_correct','correct_to_wrong']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','evaluate']);args=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'evaluate':evaluate}[args.stage]()

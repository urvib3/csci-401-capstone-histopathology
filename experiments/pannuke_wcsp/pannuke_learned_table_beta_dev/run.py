"""Reproduce the frozen development sweep; never select new IDs or run test patches."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,time
import numpy as np
from core import semantic,probabilities,solve,confusion,metrics

PACKAGE=Path(__file__).resolve().parent
REPO=next(p for p in PACKAGE.parents if (p/'benchmarks/pannuke').is_dir())
SOURCE=REPO/'experiments/pannuke_wcsp/pannuke_d5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',required=True,type=Path,help='New output directory; existing directories are refused')
 parser.add_argument('--data-root',type=Path,default=REPO/'benchmarks/pannuke/data')
 parser.add_argument('--checkpoint',required=True,type=Path)
 args=parser.parse_args()
 HERE=args.output.resolve();DATA=args.data_root.resolve();weights=args.checkpoint.resolve()
 protocol=json.loads((PACKAGE/'protocol.json').read_text())
 cfg=json.loads((SOURCE/'frozen_config.json').read_text())
 assert sha(SOURCE/'frozen_config.json')==protocol['source_config_sha256']
 assert sha(PACKAGE/'core.py')==protocol['core_sha256']
 assert sha(weights)==protocol['checkpoint_sha256']
 assert protocol['table']==cfg['table']
 BETAS=protocol['betas'];SEED=protocol['seed'];table=np.array(protocol['table']);records=protocol['development']
 ids={r['image_index'] for r in records};split=json.loads((SOURCE/'split.json').read_text())
 assert len(ids)==20 and not ids.intersection(protocol['excluded_ids'])
 assert not ids.intersection(r['image_index'] for n in ['train','test'] for r in split[n])
 masks=np.load(DATA/'masks/fold2/masks.npy',mmap_mode='r')
 types=np.load(DATA/'images/fold2/types.npy',allow_pickle=False)
 for record in records:
  i=record['image_index'];gt,_=semantic(masks[i])
  assert str(types[i])==record['tissue']
  assert np.bincount(gt.ravel(),minlength=6).tolist()==record['class_pixels']
 HERE.mkdir(parents=True,exist_ok=False)
 dump(HERE/'protocol.json',protocol)
 import torch,yaml,importlib.metadata
 pkg=Path(importlib.util.find_spec('tiatoolbox').origin).parent
 meta=yaml.safe_load((pkg/'data/pretrained_model.yaml').read_text())['hovernet_fast-pannuke']
 # Older installed TIAToolbox keeps type labels on the model class.
 start=time.perf_counter()
 from tiatoolbox.models.architecture import get_pretrained_model
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
 model,_=get_pretrained_model('hovernet_fast-pannuke',pretrained_weights=str(weights));model.eval()
 actual=model.class_dict['nuclei_segmentation'];assert [actual[k].lower() for k in range(6)]==[s.lower() for s in cfg['classes']]
 setup=time.perf_counter()-start
 dump(HERE/'environment.json',dict(model_setup_seconds=setup,channel_mapping=actual,versions={n:importlib.metadata.version(n) for n in ['numpy','numba','torch','tiatoolbox']},architecture_sha256={n:sha(pkg/'models/architecture'/n) for n in ['hovernet.py','utils.py']},note='Local TIAToolbox 2.0.1/PyTorch 2.9.1 versus D5 recorded 2.1.2/2.12.0. Same checkpoint hash, verified mapping and direct raw TP forward; environment difference recorded.'))
 images=np.load(DATA/'images/fold2/images.npy',mmap_mode='r');rows=[];cms={b:[] for b in BETAS}
 (HERE/'predictions').mkdir();solve(np.ones((3,3,6))/6,table)
 for rec in records:
  i=rec['image_index'];g,_=semantic(masks[i]);start=time.perf_counter()
  with torch.inference_mode():raw=model(torch.from_numpy(np.array(images[i],dtype=np.float32)).permute(2,0,1)[None])['tp'].cpu().numpy()
  assert raw.shape==(1,6,164,164);p=probabilities(raw[0].transpose(1,2,0));infer=time.perf_counter()-start
  a=p.argmax(-1).astype(np.uint8);saved=dict(gt=g,baseline=a);audits={}
  for beta in BETAS:
   start=time.perf_counter();y,info=solve(p,table,beta=beta);runtime=time.perf_counter()-start
   if beta==0:assert np.array_equal(a,y)
   cm=confusion(g,y);cms[beta].append(cm);m=metrics(cm)
   row=dict(image_index=i,tissue=rec['tissue'],beta=beta,biological_miou=m['biological_miou'],six_class_miou=m['six_class_miou'],pixel_accuracy=m['pixel_accuracy'],changed_pixels=int((a!=y).sum()),wrong_to_correct=int(((a!=g)&(y==g)).sum()),correct_to_wrong=int(((a==g)&(y!=g)).sum()),wrong_to_different_wrong=int(((a!=y)&(a!=g)&(y!=g)).sum()),solver_seconds=runtime,inference_seconds=infer,rounds=info['rounds'],local_minimum=info['verified_local_minimum'],stop_reason=info['stop_reason'])
   rows.append(row);saved[f'beta_{beta:g}']=y;audits[str(beta)]=info
  np.savez_compressed(HERE/f'predictions/patch_{i:04d}.npz',**saved);dump(HERE/f'predictions/patch_{i:04d}.json',audits)
  print(f'patch {i}: completed all {len(BETAS)} betas; inference {infer:.2f}s',flush=True)
 agg=[]
 for beta in BETAS:
  rr=[r for r in rows if r['beta']==beta];m=metrics(np.sum(cms[beta],0));row=dict(beta=beta,**m)
  for k in ['changed_pixels','wrong_to_correct','correct_to_wrong','wrong_to_different_wrong','solver_seconds']:row[k]=sum(r[k] for r in rr)
  row.update(local_minima=sum(r['local_minimum'] for r in rr),max_rounds=max(r['rounds'] for r in rr),round_limit_count=sum(r['stop_reason']=='round_limit' for r in rr));agg.append(row)
 best=max(agg,key=lambda r:(r['biological_miou'],-r['beta']))
 dump(HERE/'results.json',dict(aggregate=agg,selected_beta=best['beta'],selection='Development only; selection-biased, not a generalization estimate',inference_seconds=sum(r['inference_seconds'] for r in rows if r['beta']==0),model_setup_seconds=setup))
 for name,data in [('per_patch',rows),('aggregate_beta',agg)]:
  with (HERE/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 np.savez_compressed(HERE/'confusions.npz',**{f'beta_{b:g}':np.array(c) for b,c in cms.items()})
 print('SELECTED DEVELOPMENT BETA',best['beta'],flush=True)
if __name__=='__main__':main()

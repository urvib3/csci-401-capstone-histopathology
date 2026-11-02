"""NumPy-only independent verification of published sweep artifacts."""
import hashlib,json
from pathlib import Path
import numpy as np

def main():
 here=Path(__file__).resolve().parent
 cfg=json.loads((here/'protocol.json').read_text());results=json.loads((here/'results.json').read_text());selected=json.loads((here/'selected_config.json').read_text())
 d5=here.parent/'pannuke_d5'
 assert cfg['table']==json.loads((d5/'frozen_config.json').read_text())['table']==selected['table']
 assert hashlib.sha256((d5/'frozen_config.json').read_bytes()).hexdigest()==cfg['source_config_sha256']
 assert hashlib.sha256((here/'core.py').read_bytes()).hexdigest()==cfg['core_sha256']
 split=json.loads((d5/'split.json').read_text());ids={r['image_index'] for r in cfg['development']}
 assert len(ids)==20 and not ids.intersection(cfg['excluded_ids'])
 assert not ids.intersection(r['image_index'] for n in ['train','test'] for r in split[n])
 files=list((here/'predictions').glob('*.npz'));assert len(files)==20
 assert {int(f.stem.split('_')[1]) for f in files}==ids
 stored=np.load(here/'confusions.npz')
 for row in results['aggregate']:
  cm=np.zeros((6,6),int);tot=np.zeros(4,int)
  for f in files:
   with np.load(f) as z:
    g=z['gt'];a=z['baseline'];b=z[f'beta_{row["beta"]:g}'];assert g.shape==a.shape==b.shape==(164,164)
    cm+=np.bincount((g.astype(int)*6+b).ravel(),minlength=36).reshape(6,6)
    tot+=np.array([(a!=b).sum(),((a!=g)&(b==g)).sum(),((a==g)&(b!=g)).sum(),((a!=b)&(a!=g)&(b!=g)).sum()])
    if row['beta']==0:assert np.array_equal(a,b)
   audit=json.loads(f.with_suffix('.json').read_text())[str(row['beta'])]
   assert audit['verified_local_minimum'] and audit['remaining_improving_pixels']==0 and audit['stop_reason']=='no_changes'
   assert np.all(np.diff(audit['energy'])<=1e-7)
  assert np.array_equal(cm,stored[f'beta_{row["beta"]:g}'].sum(0))
  union=cm.sum(0)+cm.sum(1)-np.diag(cm);iou=np.divide(np.diag(cm),union,out=np.zeros(6,float),where=union>0)
  assert np.allclose(iou,row['iou'],rtol=0,atol=1e-14)
  assert abs(iou[1:].mean()-row['biological_miou'])<1e-14
  assert abs(iou.mean()-row['six_class_miou'])<1e-14
  assert abs(np.trace(cm)/cm.sum()-row['pixel_accuracy'])<1e-14
  assert tot.tolist()==[row[k] for k in ['changed_pixels','wrong_to_correct','correct_to_wrong','wrong_to_different_wrong']]
 best=max(results['aggregate'],key=lambda r:(r['biological_miou'],-r['beta']))
 assert best['beta']==results['selected_beta']==selected['beta']==.15
 print('PASS: frozen table, split exclusions, 300 prediction recounts, beta-zero identity, recorded solver audits, and selected beta 0.15.')

if __name__=='__main__':main()

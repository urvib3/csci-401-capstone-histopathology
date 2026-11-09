"""Recount outputs, save paired-bootstrap draws, verify preservation."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from core import confusion,metrics
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'hierarchy_20261109';ROOT=HERE.parents[2]
def main():
    proto=json.loads((HERE/'protocol.json').read_text());comparison=json.loads((HERE/'comparisons.json').read_text());arrays={};verified=[]
    for name in ['scaled_matched','scaled_calibrated','parent_only','potts','biological']:
        r=json.loads((HERE/name/'results.json').read_text());aa=[];bb=[];fix=brk=chg=other=0
        for row in r['patches']:
            i=row['image_index'];gt=np.load(PREV/f'cache/gt_{i:04d}.npy');raw=np.load(PREV/f'cache/prob_{i:04d}.npz')['p'].argmax(-1);pred=np.load(HERE/name/f'patch_{i:04d}.npz')['pred'];aa.append(confusion(gt,raw));bb.append(confusion(gt,pred))
            f=(raw!=gt)&(pred==gt);b=(raw==gt)&(pred!=gt);c=raw!=pred
            fix+=int(f.sum());brk+=int(b.sum());chg+=int(c.sum());other+=int((c&~f&~b).sum());assert np.all(np.diff(row['solver']['energy'])<=1e-7)
        assert (fix,brk,chg,other)==tuple(r[k] for k in ['fixes','breaks','changed','different_wrong'])
        np.testing.assert_allclose(metrics(sum(bb))['iou'],r['wcsp']['iou']);arrays[name]=np.array(bb);arrays['raw']=np.array(aa);verified.append(name)
    arrays['unscaled_block_unaries']=np.load(PREV/'version_2/confusions.npz')['wcsp'];draws={}
    for key,value in comparison.items():
        a,b=key.split(' minus ');x=arrays[b];y=arrays[a];rng=np.random.default_rng(20261109);ds=[]
        for _ in range(5000):
            ix=rng.integers(len(x),size=len(x));ds.append(metrics(y[ix].sum(0))['biological_miou']-metrics(x[ix].sum(0))['biological_miou'])
        ds=np.array(ds);np.testing.assert_allclose(np.percentile(ds,[2.5,97.5]),value['ci95'],atol=1e-12);draws[key.replace(' ','_')]=ds
    np.savez_compressed(HERE/'bootstrap_replicates.npz',**draws)
    for name,digest in proto['preserve_before'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in proto['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    for link in re.findall(r'\]\(([^)]+)\)',(HERE/'REPORT.md').read_text()):assert (HERE/link).exists(),link
    report=dict(conditions_verified=verified,bootstrap_draws_verified=True,preserved_original_experiment=True,deliverable_6_unchanged=True,source_hashes=True,report_links=True)
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

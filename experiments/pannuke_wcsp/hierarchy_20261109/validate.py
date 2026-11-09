"""Validate saved experiment data/results without rerunning optimization."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from core import metrics,confusion,normalize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    cfg=json.loads((HERE/'frozen_config.json').read_text());split=json.loads((HERE/'split.json').read_text());ex=json.loads((HERE/'exclusions.json').read_text());tabs=json.loads((HERE/'tables.json').read_text());inf=json.loads((HERE/'inference.json').read_text())
    sets={k:{r['image_index'] for r in split[k]} for k in ['train','dev','test']}
    assert [len(sets[k]) for k in ['train','dev','test']]==[200,12,100]
    assert not sets['train']&sets['dev'] and not sets['train']&sets['test'] and not sets['dev']&sets['test']
    assert not {int(x) for x in ex}&(sets['dev']|sets['test'])
    assert sha(HERE/'split.json')==cfg['split_sha256'] and sha(HERE/'tables.json')==cfg['tables_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(HERE/name)==digest
    old=json.loads((HERE.parent/'pannuke_d5/frozen_config.json').read_text())
    assert inf['checkpoint_sha256']==old['checkpoint']['sha256'] and inf['architecture_sha256']==old['architecture_source_sha256']
    for tab,ev in zip(tabs['final'],tabs['evidence']):
        tab=np.array(tab);data=np.array(ev['learned']);prior=np.array(ev['prior'])
        np.testing.assert_allclose(tab,normalize(.8*data+.2*prior));np.testing.assert_allclose(prior[0],data[0]);np.testing.assert_allclose(tab,tab.T)
    outputs=[]
    for version in [1,2]:
        dest=HERE/f'version_{version}'
        if not (dest/'results.json').exists():continue
        result=json.loads((dest/'results.json').read_text());aa=[];bb=[];changed=fixes=breaks=other=0
        for row in result['patches']:
            i=row['image_index'];a=np.load(dest/f'patch_{i:04d}.npz');p=np.load(HERE/f'cache/prob_{i:04d}.npz')['p']
            assert p.shape==(256,256,6) and np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6)
            np.testing.assert_array_equal(a['raw'],p.argmax(-1));np.testing.assert_array_equal(a['gt'],np.load(HERE/f'cache/gt_{i:04d}.npy'))
            aa.append(confusion(a['gt'],a['raw']));bb.append(confusion(a['gt'],a['pred']))
            ch=a['raw']!=a['pred'];fx=(a['raw']!=a['gt'])&(a['pred']==a['gt']);br=(a['raw']==a['gt'])&(a['pred']!=a['gt'])
            changed+=int(ch.sum());fixes+=int(fx.sum());breaks+=int(br.sum());other+=int((ch&~fx&~br).sum())
            assert np.all(np.diff(row['solver']['energy'])<=1e-7)
        for key,arr in [('raw',aa),('wcsp',bb)]:
            m=metrics(sum(arr));np.testing.assert_allclose(m['iou'],result[key]['iou'])
            assert abs(m['biological_miou']-result[key]['biological_miou'])<1e-12
        assert (changed,fixes,breaks,other)==tuple(result[k] for k in ['changed','fixes','breaks','different_wrong'])
        assert changed==fixes+breaks+other
        outputs.append(dict(version=version,patches=len(aa),metrics_verified=True,changed=changed,fixes=fixes,breaks=breaks))
    doc=ROOT/'deliverables/deliverable_6.md'
    if doc.exists():
        text=doc.read_text()
        if '<!-- BEGIN PANNUKE HIERARCHY 20261109 -->' in text:
            text=text.split('<!-- BEGIN PANNUKE HIERARCHY 20261109 -->',1)[1].split('<!-- END PANNUKE HIERARCHY 20261109 -->',1)[0]
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith('http'):assert (doc.parent/target).resolve().exists(),target
    overlap=json.loads((HERE/'upstream_overlap_audit.json').read_text()) if (HERE/'upstream_overlap_audit.json').exists() else None
    report=dict(split_separation=True,previous_use_exclusion_against_original_snapshot=True,upstream_overlap_audit=overlap,source_hashes=True,checkpoint_architecture=True,tables_reconstructed=True,outputs=outputs,markdown_links=True)
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

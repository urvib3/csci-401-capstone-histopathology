"""Reproduce this exact frozen split/table in a NEW sibling directory; never resample."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from core import semantic,adjacency,fit
from run import stream_arrays,sha,DATA
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    out=args.output.resolve()
    if out.exists():p.error('Output must be new; existing experiment files are preserved.')
    if out.parent!=HERE.parent:p.error('Use a new sibling directory under experiments/.')
    cfg=json.loads((HERE/'frozen_config.json').read_text());split=json.loads((HERE/'split.json').read_text());tables=json.loads((HERE/'tables.json').read_text())
    assert sha(DATA/'fold_2.zip')==cfg['archive']['sha256']
    assert sha(HERE/'split.json')==cfg['split_sha256']
    assert all(sha(HERE/n)==v for n,v in cfg['provenance']['source_hashes'].items())
    out.mkdir();(out/'cache').mkdir()
    names=['core.py','run.py','test_experiment.py','validate_inputs.py','make_figures.py','make_report.py','make_summary.py','deck_builder.mjs','make_deck.mjs','reproduce.py','.gitignore','README.md','split.json','coverage.json','train_manifest.csv','test_manifest.csv','previous_use_exclusions.json','tables.json','frozen_config.json','frozen_config.sha256']
    for name in names:shutil.copy2(HERE/name,out/name)
    for file in HERE.glob('table_*.csv'):shutil.copy2(file,out/file.name)
    chosen={r['image_index'] for part in ['train','test'] for r in split[part]}
    # Populate only the fixed 220 selected entries; other cache rows are unused.
    gt=np.lib.format.open_memmap(out/'cache/all_semantic.npy',mode='w+',dtype=np.uint8,shape=(2523,164,164))
    gt[:]=0
    for i,mask in stream_arrays('masks.npy',chosen):gt[i]=semantic(mask)[0]
    gt.flush()
    counts=sum((adjacency(gt[r['image_index']]) for r in split['train']),start=np.zeros((6,6),np.int64))
    np.testing.assert_array_equal(counts,np.array(tables['counts']))
    np.testing.assert_allclose(fit(counts)['final'],np.array(cfg['table']),rtol=0,atol=1e-14)
    ids={r['image_index'] for r in split['test']}
    for i,image in stream_arrays('images.npy',ids):
        assert np.array_equal(image,image.astype(np.uint8))
        np.save(out/f'cache/rgb_{i:04d}.npy',image.astype(np.uint8))
    (out/'reproduction_note.json').write_text(json.dumps(dict(original=str(HERE),frozen_config_sha256=sha(HERE/'frozen_config.json'),exact_manifests_reused=True,training_table_recomputed_and_matched=True,unused_cache_rows='zeros; never used'),indent=2)+'\n')
    if not args.prepare_only:subprocess.run([sys.executable,str(out/'run.py'),'evaluate'],check=True)
    print(out)

if __name__=='__main__':main()

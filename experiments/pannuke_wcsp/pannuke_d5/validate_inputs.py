"""Independent real-input alignment and lossless RGB validation; no model results used."""
import json
from pathlib import Path
import numpy as np
from run import stream_arrays,sha
HERE=Path(__file__).resolve().parent
split=json.loads((HERE/'split.json').read_text());ids=[r['image_index'] for r in split['test']]
gt=np.load(HERE/'cache/all_semantic.npy',mmap_mode='r');rows={i:{} for i in ids}
for i,m in stream_arrays('masks.npy',ids):
    assert m.shape==(256,256,6)
    occ=m[46:210,46:210,:5]>0
    independent=(occ*np.arange(1,6)).max(-1).astype(np.uint8)
    assert np.array_equal(gt[i],independent)
    rows[i]['gt_crop_exact']=True
    rows[i]['gt_pixels']=np.bincount(independent.ravel(),minlength=6).tolist()
for i,image in stream_arrays('images.npy',ids):
    cache=np.load(HERE/f'cache/rgb_{i:04d}.npy')
    assert image.shape==(256,256,3) and np.isfinite(image).all()
    assert np.array_equal(image,cache),'RGB input must be lossless, without quantization'
    rows[i]['rgb_lossless']=True
(HERE/'input_validation.json').write_text(json.dumps(dict(split_sha256=sha(HERE/'split.json'),patches=rows,gt_crop=[46,210,46,210]),indent=2)+'\n')
print('All 20 test RGB inputs are lossless and GT crops exactly aligned.')

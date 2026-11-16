# Shared full-resolution BCSS development inputs

This directory publishes the two inputs needed to reproduce the paired
full-resolution development experiments from Ryan's cluster checkout:

- **`inputs/`**: 185 `.npz` files containing the U-Net probabilities, ground
  truth and tile ID. Binary inputs are stored with **Git LFS**.
- **`protocol.json`**: the exact tile list, model/source/configuration hashes,
  original experiment settings and baseline metrics, stored as ordinary JSON.

These are **development inputs**, not held-out validation data. They are the
185 scored whole-field tiles from the evenly spaced 200-tile development
selection, spanning 14 ROIs in preparation version 2. The 15 entirely ignored
tiles in that selection are absent. No images, other datasets, probability
caches, training checkpoints or result arrays are included in this bundle.

## Download and check

From the repository root with Git LFS installed:

```bash
git lfs install
git lfs pull --include="data/fullres_multiscale_20261115/inputs/*.npz" --exclude=""
cd data/fullres_multiscale_20261115
sha256sum -c SHA256SUMS
```

The broad `/data/` ignore rule is preserved. Only this explicitly requested
bundle is tracked. A normal clone with Git LFS enabled downloads the arrays
automatically; otherwise the `.npz` paths initially hold LFS text pointers.
On the current cluster checkout, `data/tools/git-lfs` is the local executable;
`export PATH="$PWD/data/tools:$PATH"` makes it available to Git.

## Array contents

```python
import json
import numpy as np
from pathlib import Path

root = Path("data/fullres_multiscale_20261115")
protocol = json.loads((root / "protocol.json").read_text())
record = protocol["samples"][0]
with np.load(root / "inputs" / f"{record['index']:04d}.npz") as tile:
    probabilities = tile["probs"]  # float16, (5, 512, 512)
    ground_truth = tile["target"]   # uint8, (512, 512); 255 means ignored
    tile_id = str(tile["id"])
```

Class IDs/channels follow WCSP order: 0 neoplastic (tumor), 1 inflammatory,
2 connective (stroma), 3 dead (necrosis), 4 healthy (BCSS "other"). The last
class includes fat, vessels and other tissue rather than only healthy
epithelium. The 512x512 field is the whole tile, with no central crop.
Cast probabilities to float32 before building the WCSP to match the runner.
Score only pixels whose target is not 255. The pooled argmax mIoU is
**0.6524144074620204** on this frozen population.

`SHA256SUMS` covers the unchanged original protocol and every input file;
the per-input hashes also appear in `protocol.json`. Keep that protocol
unchanged as provenance. Its `files` section references the original source
tree and model checkpoint, which are separate artifacts. Loading these
cached inputs and testing a new heuristic does not require model inference;
the original runner's strict `verify()` additionally checks those referenced
files if you use it unchanged.

See the [development runbook](../../experiments/bcss_wcsp/multiscale/FULLRES.md)
and [runner](../../experiments/bcss_wcsp/multiscale/fullres_sweep.py). Write new
experiment outputs to a separate directory so the frozen bundle is preserved.

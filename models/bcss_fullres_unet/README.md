# BCSS full-resolution U-Net checkpoint

The selected trained model is **[`best.pt`](best.pt)** in this directory.
Git LFS stores the checkpoint contents; an ordinary Git blob contains only
the LFS pointer. The file is 147,627,952 bytes (140.79 MiB). Its SHA256 is:

```text
9cceca988b77f0b08b276a7dac0d8ee83e9574e922fe93bfdddb8d6759fd8e17
```

## Download and verify

Install Git LFS on your machine, then run from the repository root:

```bash
git lfs install
git lfs pull --include="models/bcss_fullres_unet/best.pt" --exclude=""
cd models/bcss_fullres_unet
sha256sum -c SHA256SUMS
```

If `best.pt` starts with `version https://git-lfs.github.com/spec/v1`, it
is still a pointer; run the LFS pull before loading it with PyTorch.
On Ryan's current cluster checkout, a repo-local Git LFS executable is at
`data/tools/git-lfs`; `export PATH="$PWD/data/tools:$PATH"` makes it available
for Git commands. That executable is local tooling and is not committed.

## Model and training

This is the TIAToolbox BCSS ResNet-50 U-Net with a trained full-resolution
RGB encoder/decoder refinement head (width 32). It predicts five logits for
every pixel in a 512x512 RGB tile in one forward pass. Both the pretrained
backbone and the refinement head were fine-tuned using cross-entropy.
The checkpoint contains both backbone and head weights. The optimizer/resume
checkpoint `last.pt` is not included; `best.pt` is the selected inference model.

- Training: [`train_experiments.py`](../../benchmarks/bcss/scripts/train_experiments.py),
  architecture `fullres`, field `full`, pairwise weight 0, 10 epochs,
  batch 4, seed 0, backbone learning rate 1e-4, head learning rate 1e-3.
- Selected checkpoint: epoch 7 by whole-field dev mIoU on all 1,012 dev tiles.
  Whole-field mIoU 0.664832; central-256 mIoU 0.679658.
- [`settings.json`](settings.json) preserves the original training settings,
  including cluster-specific source/output paths; [`log.csv`](log.csv)
  preserves the epoch metrics. Adapt absolute paths when reproducing training.
- Data: preparation version 2, 0.25 MPP, 7,768 fit / 1,012 dev / 4,028 val
  tiles from the 151 BCSS ROIs, hospital-held-out train/val split.
  These are development-selected weights; no new held-out validation score
  is claimed for this model.
- Native model channels: tumor, stroma, inflammatory, necrosis, other.
  The WCSP adapter reorders probabilities to neoplastic, inflammatory,
  connective, dead, healthy (the last label represents BCSS "other").

## Load or use with experiment configs

The loader is [`FullResTiaUNet`](../../wcsp/classifiers/tia_unet.py), backed by
[`FullResUNet`](../../wcsp/classifiers/fullres_unet.py). Use the repository's
training dependencies in [`requirements-training.txt`](../../experiments/bcss_wcsp/requirements-training.txt).

```python
import numpy as np
from PIL import Image
from wcsp.classifiers.tia_unet import FullResTiaUNet
from wcsp.data.base import Sample

model = FullResTiaUNet(
    checkpoint="models/bcss_fullres_unet/best.pt",
    device="cpu",  # or "cuda"
    threads=4,
)
image = np.asarray(Image.open("tile.png").convert("RGB"))
assert image.shape == (512, 512, 3)
prediction = model.predict([Sample(id="tile", image=image, target=None)])[0]
labels = prediction.probs.argmax(axis=0)  # (512, 512), WCSP label order
```

The current loader first constructs the TIAToolbox base model, downloading
its official pretrained weights if they are not cached, then loads this
fine-tuned checkpoint. Its optional `weights=` argument can point to an
existing `fcn_resnet50_unet-bcss.pth` to avoid that download.

The running [full-resolution multiscale sweep](../../experiments/bcss_wcsp/multiscale/FULLRES.md)
uses the byte-identical original local file at
`data/outputs/tonight/full512/best.pt`. That directory is ignored by Git.
To use unchanged experiment configs in a new checkout, restore that path:

```bash
mkdir -p data/outputs/tonight/full512
cp models/bcss_fullres_unet/best.pt data/outputs/tonight/full512/best.pt
```

Alternatively, pass `--set classifier.checkpoint=models/bcss_fullres_unet/best.pt`
to the WCSP CLI. Dataset files and cached predictions remain separate local
artifacts; publishing the checkpoint does not include those caches.

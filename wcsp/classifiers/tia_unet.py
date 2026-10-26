# Pretrained TIAToolbox BCSS U-Net (fcn_resnet50_unet-bcss) as a pixel classifier.
#
# Same protocol as benchmarks/bcss/scripts/evaluate_pretrained.py: the
# network sees a 512 x 512 tile, its half-resolution softmax is upsampled
# x2, and only the central 256 x 256 is kept. Output classes are in BCSS
# coarse order and are reordered to wcsp order here.
#
# native=True replaces the x2 bilinear upsampling with shift-and-stitch: the
# network runs on the tile shifted by (0|1, 0|1) px, and the four
# half-resolution outputs are interleaved into a full-resolution map, so
# every pixel gets its own network prediction (4x the compute). The
# network's output i sits on input pixel 2i (stride-2 stem, padding 3), so
# the copy shifted by s fills pixels 2i + s.

import numpy as np

from ..labels import BCSS_TO_WCSP, reorder_probs
from .base import Prediction

MODEL_NAME = "fcn_resnet50_unet-bcss"
TILE = 512
CROP = 256
OFFSET = (TILE - CROP) // 2
SHIFTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def shift_stitch(forward, x):
    """
    Full-resolution output of a stride-2 network.

    forward:
        (N, C, H, W) -> (N, K, H/2, W/2).
    x:
        (N, C, H, W) input; each shifted copy drops its first s rows / cols
        and reflects s at the far edge, so only the tile border changes.
    """
    import torch.nn.functional as F

    out = None
    for sy, sx in SHIFTS:
        shifted = F.pad(x[..., sy:, sx:], (0, sx, 0, sy), mode="reflect") if sy or sx else x
        y = forward(shifted)
        if out is None:
            out = y.new_empty(y.shape[:2] + x.shape[2:])
        out[..., sy::2, sx::2] = y
    return out


class TiaUNet:
    """
    weights:
        Optional local checkpoint path; default downloads TIAToolbox's.
    native:
        Shift-and-stitch full-resolution output instead of x2 bilinear.
    """

    def __init__(self, device="cpu", threads=None, weights=None, native=False):
        import torch
        from tiatoolbox.models.architecture import get_pretrained_model

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.native = native
        self.name = "tia_unet_native" if native else "tia_unet"
        model, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=weights)
        self.model = model.eval().to(device)

    def _forward(self, x):
        return self.torch.nn.functional.softmax(self.model(self.model.preproc(x)), dim=1)

    def _full_probs(self, x):
        """(N, K, TILE, TILE) BCSS-order probabilities for a batch of tiles."""
        if self.native:
            return shift_stitch(self._forward, x)
        F = self.torch.nn.functional
        return F.interpolate(self._forward(x), scale_factor=2, mode="bilinear", align_corners=False)

    def predict(self, samples):
        torch = self.torch
        images = np.stack([s.image for s in samples])
        if images.shape[1:3] != (TILE, TILE):
            raise ValueError(f"{self.name} expects {TILE}x{TILE} tiles.")
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().to(self.device)
        with torch.inference_mode():
            probs = self._full_probs(x)[..., OFFSET:OFFSET + CROP, OFFSET:OFFSET + CROP].cpu().numpy()
        return [Prediction(reorder_probs(p, BCSS_TO_WCSP, axis=0)) for p in probs]


class FullResTiaUNet(TiaUNet):
    """
    The U-Net with a trained full-resolution level (wcsp/classifiers/fullres_unet.py).

    checkpoint:
        Output of benchmarks/bcss/scripts/train_fullres_unet.py.
    """

    name = "tia_unet_fullres"

    def __init__(self, checkpoint, device="cpu", threads=None, weights=None):
        import torch

        from .fullres_unet import load_fullres_unet

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.native = False
        self.model = load_fullres_unet(checkpoint, weights=weights, device=device)

    def _full_probs(self, x):
        return self.torch.nn.functional.softmax(self.model(x), dim=1)

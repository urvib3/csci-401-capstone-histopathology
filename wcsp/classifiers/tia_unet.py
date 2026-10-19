# Pretrained TIAToolbox BCSS U-Net (fcn_resnet50_unet-bcss) as a pixel classifier.
#
# Same protocol as benchmarks/bcss/scripts/evaluate_pretrained.py: the
# network sees a 512 x 512 tile, its half-resolution softmax is upsampled
# x2, and only the central 256 x 256 is kept. Output classes are in BCSS
# coarse order and are reordered to wcsp order here.

import numpy as np

from ..labels import BCSS_TO_WCSP, reorder_probs
from .base import Prediction

MODEL_NAME = "fcn_resnet50_unet-bcss"
TILE = 512
CROP = 256
OFFSET = (TILE - CROP) // 2


class TiaUNet:
    name = "tia_unet"

    def __init__(self, device="cpu", threads=None):
        import torch
        from tiatoolbox.models.architecture import get_pretrained_model

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        model, _ = get_pretrained_model(MODEL_NAME)
        self.model = model.eval().to(device)

    def predict(self, samples):
        torch = self.torch
        F = torch.nn.functional
        images = np.stack([s.image for s in samples])
        if images.shape[1:3] != (TILE, TILE):
            raise ValueError(f"{self.name} expects {TILE}x{TILE} tiles.")
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
        with torch.inference_mode():
            logits = self.model(self.model.preproc(x).to(self.device))
            probs = F.softmax(logits, dim=1)
            probs = F.interpolate(probs, scale_factor=2, mode="bilinear", align_corners=False)
            probs = probs[..., OFFSET:OFFSET + CROP, OFFSET:OFFSET + CROP].cpu().numpy()
        return [Prediction(reorder_probs(p, BCSS_TO_WCSP, axis=0)) for p in probs]

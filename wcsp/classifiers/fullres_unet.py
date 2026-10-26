# The pretrained TIAToolbox BCSS U-Net with one more encoder / decoder level,
# so it predicts at full resolution instead of half.
#
#   input 512 ──► full-res encoder (2 conv, 32 ch) ─────────────┐ skip
#     │                                                          ▼
#     └──► pretrained U-Net ─► decoder features (64 ch, 256) ─► up x2 ─► concat ─► 2 conv ─► 1x1 ─► + ─► logits (512)
#                          └─► pretrained logits (256) ───────► bilinear x2 ──────────────────────────┘
#
# The new head predicts a correction to the pretrained logits, upsampled x2,
# and its last layer starts at zero, so an untrained model reproduces the
# pretrained U-Net (with bilinear upsampling of logits). Training only has to
# learn what full resolution adds. Output classes are in BCSS coarse order.

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "fcn_resnet50_unet-bcss"


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FullResUNet(nn.Module):
    """
    base:
        A TIAToolbox UNetModel; its weights are kept, and frozen when
        freeze_base is set.
    width:
        Channels of the new full-resolution layers.
    """

    def __init__(self, base, width=32, freeze_base=True):
        super().__init__()
        self.base = base
        self.freeze_base = freeze_base
        if freeze_base:
            base.requires_grad_(False)
        self.encoder = nn.Sequential(conv_block(3, width), conv_block(width, width))
        decoder_channels = base.clf.in_channels   # 64 for the BCSS model
        self.decoder = nn.Sequential(conv_block(decoder_channels + width, width), conv_block(width, width))
        self.clf = nn.Conv2d(width, base.clf.out_channels, 1)
        nn.init.zeros_(self.clf.weight)
        nn.init.zeros_(self.clf.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()   # keep frozen batch-norm statistics
        return self

    def _base_features(self, imgs):
        """The pretrained forward pass, also returning the features before its classifier."""
        base = self.base
        en_list = base.backbone(base._transform(imgs))
        x = base.conv1x1(en_list[-1])
        en_list = en_list[:-1]
        for idx in range(1, len(en_list) + 1):
            y = en_list[-idx]
            x_ = base.upsample2x(x)
            x = x_ + y if base.skip_type == "add" else torch.cat([x_, y], dim=1)
            x = base.uplist[idx - 1](x)
        return x, base.clf(x)

    def forward(self, imgs):
        """
        imgs:
            (N, 3, H, W) float RGB in 0-255, like the pretrained model.

        returns:
            (N, K, H, W) logits.
        """
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_base):
            features, logits = self._base_features(imgs)
        size = imgs.shape[-2:]
        skip = self.encoder(imgs / 255.0)
        up = F.interpolate(features, size=size, mode="bilinear", align_corners=False)
        correction = self.clf(self.decoder(torch.cat([up, skip], dim=1)))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False) + correction

    def head_state_dict(self):
        """Only the new layers; the base comes from the TIAToolbox weights."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base.")}


def load_fullres_unet(checkpoint=None, weights=None, width=32, device="cpu"):
    """
    checkpoint:
        A file saved by benchmarks/bcss/scripts/train_fullres_unet.py, or None
        for the untrained head. Holds the head, plus the base when it was
        fine-tuned.
    weights:
        Optional local copy of the TIAToolbox weights.
    """
    from tiatoolbox.models.architecture import get_pretrained_model

    base, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=weights)
    state = torch.load(checkpoint, map_location="cpu") if checkpoint else None
    if state is not None:
        width = state.get("width", width)
    model = FullResUNet(base, width=width)
    if state is not None:
        model.load_state_dict(state["model"], strict=False)
        missing = set(model.head_state_dict()) - set(state["model"])
        if missing:
            raise ValueError(f"{checkpoint} lacks {sorted(missing)}")
    return model.eval().to(device)

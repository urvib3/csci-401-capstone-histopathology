# A pathology foundation model + a light segmentation decoder, fine-tuned on BCSS (fit
# split), on the same 256 x 256 half-resolution grid as the U-Net, so the outputs drop
# straight into WCSP. Only fully open weights (no access request):
#
#   kaiko-ai/midnight   Midnight-12k: ViT-g/14 (~1.1B params, 40 layers, 1536-d), DINOv2
#                       self-supervised on 12k public TCGA slides (MIT). Default.
#   owkin/phikon-v2     ViT-L/16 (~300M params, 24 layers, 1024-d), DINOv2 on 460M tiles.
#
# The encoder turns the tile into one feature vector per patch (14 or 16 px). Decoder
# (ours, ~8M params): features from 4 depths (1/4, 1/2, 3/4, last) are projected and
# fused at patch resolution, upsampled in 3 conv stages, and joined with a small
# full-resolution conv branch on the raw image so borders are not limited to patches.
# Output: per-pixel logits for the 5 BCSS classes.
#
# Training: cross-entropy + Dice (ignore 255), AdamW, cosine schedule; encoder frozen
# except its last --unfreeze layers (lower lr). Model selection on dev mIoU. At the end,
# probabilities for dev and val (8-view flip/rotation averaging) are written to the WCSP
# cache under --tag, so `python -m wcsp solve` can run WCSP / the presence rule on top.
#
#   python experiments/bcss_wcsp/fm/train_fm.py --encoder kaiko-ai/midnight --unfreeze 4

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.classifiers.base import Prediction  # noqa: E402
from wcsp.classifiers.cache import ProbStore  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

NORM = {   # per-model input normalization (model cards)
    "kaiko-ai/midnight": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "owkin/phikon-v2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class Encoder(nn.Module):
    """HuggingFace DINOv2 backbone -> patch features after 4 layers."""

    def __init__(self, name, unfreeze=0):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        cfg = self.model.config
        self.patch, self.dim, depth = cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers
        self.taps = [depth // 4, depth // 2, 3 * depth // 4, depth]    # hidden_states indices (0 = embeddings)
        self.skip = 1 + (getattr(cfg, "num_register_tokens", None) or 0)
        mean, std = NORM[name]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        for p in self.model.parameters():
            p.requires_grad = False
        self.trainable = list(range(depth - unfreeze, depth))
        for i in self.trainable:
            for p in self.model.encoder.layer[i].parameters():
                p.requires_grad = True

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1], H and W multiples of the patch size."""
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        out = self.model(pixel_values=(x - self.mean) / self.std, output_hidden_states=True)
        states = out.hidden_states
        feats = []
        for t in self.taps:
            s = out.last_hidden_state if t == len(states) - 1 else states[t]
            feats.append(s[:, self.skip:].transpose(1, 2).reshape(x.shape[0], -1, h, w))
        return feats


class FMSeg(nn.Module):
    def __init__(self, encoder, width=256):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(encoder.dim, width, 1), nn.BatchNorm2d(width), nn.GELU())
                                   for _ in encoder.taps])
        self.fuse = conv(width * len(encoder.taps), width)
        self.up = nn.ModuleList([conv(width, 128), conv(128, 64), conv(64, 32)])
        self.stem = nn.Sequential(conv(3, 32), conv(32, 32))      # full-resolution image branch
        self.head = nn.Sequential(conv(64, 32), nn.Conv2d(32, NUM_LABELS, 1))

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1], H and W multiples of the patch size."""
        y = self.fuse(torch.cat([p(f.float()) for p, f in zip(self.proj, self.encoder(x))], 1))
        for stage in self.up:
            y = stage(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, self.stem(x)], 1))


def load_split(cfg, split, cache_dir):
    """All labeled tiles of a split as uint8 arrays (cached as one npz)."""
    path = Path(cache_dir) / f"bcss_grid_{split}.npz"
    if path.exists():
        z = np.load(path)
        return z["ids"].tolist(), z["images"], z["targets"]
    data = pipeline.dataset_for(cfg, split)
    ids, images, targets = [], [], []
    for i in range(len(data)):
        s = data[i]
        if s.grid_target is None or not (s.grid_target != IGNORE).any():
            continue
        ids.append(s.id)
        images.append(s.grid_image[..., :3])
        targets.append(s.grid_target.astype(np.uint8))
        if len(ids) % 500 == 0:
            print(f"  {split}: {len(ids)} tiles", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), images=np.stack(images), targets=np.stack(targets))
    return ids, np.stack(images), np.stack(targets)


def to_input(images, device):
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [0, 1] (the encoder normalizes)."""
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255


def pad_to(x, patch):
    """Reflect-pad (B, C, H, W) to multiples of patch; returns padded tensor and crop slices."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % patch, (-w) % patch
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect")
    return x, (slice(ph // 2, ph // 2 + h), slice(pw // 2, pw // 2 + w))


def augment(images, targets, rng, crop):
    """Random crop (a multiple of the patch size), flips and 90-degree rotations."""
    n, h, w = targets.shape
    out_i, out_t = [], []
    for k in range(n):
        y, x = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
        im, tg = images[k, y:y + crop, x:x + crop], targets[k, y:y + crop, x:x + crop]
        r = rng.integers(4)
        im, tg = np.rot90(im, r), np.rot90(tg, r)
        if rng.random() < 0.5:
            im, tg = im[:, ::-1], tg[:, ::-1]
        out_i.append(np.ascontiguousarray(im))
        out_t.append(np.ascontiguousarray(tg))
    return np.stack(out_i), np.stack(out_t)


def loss_fn(logits, target):
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
    valid = (target != IGNORE).unsqueeze(1)
    probs = logits.softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=NUM_LABELS - 1), NUM_LABELS).permute(0, 3, 1, 2) * valid
    inter = (probs * onehot).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1)
    return ce + dice.mean()


@torch.inference_mode()
def predict(model, images, device, batch=16, tta=True):
    """(N, 5, H, W) float32 probabilities; optional flip/rotation test-time averaging."""
    model.eval()
    out = []
    for b in range(0, len(images), batch):
        x, crop = pad_to(to_input(images[b:b + batch], device), model.encoder.patch)
        views = [(r, f) for r in range(4) for f in (False, True)] if tta else [(0, False)]
        acc = 0
        for r, f in views:
            v = torch.rot90(x, r, (2, 3))
            v = v.flip(3) if f else v
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(v).float().softmax(1)
            p = p.flip(3) if f else p
            acc = acc + torch.rot90(p, -r, (2, 3))
        out.append((acc / len(views))[:, :, crop[0], crop[1]].cpu().numpy())
    return np.concatenate(out)


def evaluate(model, images, targets, device, tta=False):
    probs = predict(model, images, device, tta=tta)
    conf = sum(pixel_confusion(p.argmax(0), t) for p, t in zip(probs, targets))
    return pixel_report(conf), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--unfreeze", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--encoder", default="kaiko-ai/midnight", choices=sorted(NORM))
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/fm")
    parser.add_argument("--tag", default=None, help="cache/run name (default: from --encoder)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    cfg = load_config(args.config)
    args.tag = args.tag or args.encoder.split("/")[-1].replace("-", "") + f"_seg_u{args.unfreeze}"
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tiles = "/scratch1/ubhuwani/wcsp/uni2"          # tile caches built once (any encoder)
    fit_ids, fit_x, fit_y = load_split(cfg, "fit", tiles)
    dev_ids, dev_x, dev_y = load_split(cfg, "dev", tiles)
    print(f"fit {len(fit_ids)} tiles, dev {len(dev_ids)} tiles", flush=True)

    model = FMSeg(Encoder(args.encoder, unfreeze=args.unfreeze)).to(device)
    crop = (256 // model.encoder.patch) * model.encoder.patch   # 252 for /14, 256 for /16
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(fit_ids) / args.batch)
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    print(f"trainable: decoder {sum(p.numel() for p in dec_params) / 1e6:.1f}M, "
          f"encoder {sum(p.numel() for p in enc_params) / 1e6:.1f}M; {steps} steps", flush=True)

    best, log = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()       # encoder keeps inference behaviour (no dropout / drop path)
        order = rng.permutation(len(fit_ids))
        start, total = time.time(), 0.0
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            im, tg = augment(fit_x[idx], fit_y[idx], rng, crop)
            x = to_input(im, device)
            y = torch.from_numpy(tg).to(device).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        report, _ = evaluate(model, dev_x, dev_y, device)
        row = {"epoch": epoch + 1, "loss": total / len(order), "dev_mIoU": report["mIoU"],
               "minutes": (time.time() - start) / 60}
        log.append(row)
        print(json.dumps(row), flush=True)
        if report["mIoU"] > best:
            best = report["mIoU"]
            keep = ("encoder.model.encoder.layer.%d." % i for i in model.encoder.trainable)
            prefixes = tuple(keep)
            torch.save({k: v for k, v in model.state_dict().items()
                        if not k.startswith("encoder.") or k.startswith(prefixes)}, out / "best.pt")
        (out / "log.json").write_text(json.dumps(log, indent=2))

    # best checkpoint -> dev/val probabilities (with TTA) into the WCSP cache
    state = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    results = {"best_dev_mIoU_no_tta": best}
    for split in ("dev", "val"):
        ids, xs, ys = (dev_ids, dev_x, dev_y) if split == "dev" else load_split(cfg, "val", tiles)
        report, probs = evaluate(model, xs, ys, device, tta=True)
        results[f"{split}_tta"] = report
        store = ProbStore(cfg.cache_root, cfg.dataset["name"], args.tag, split)
        store.write(0, 1, ids, [Prediction(p) for p in probs])
        print(split, "TTA mIoU", round(report["mIoU"], 4), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Structure losses for BCSS training (train_experiments.py), in BCSS class order.

cluster_consistency
    Pixels of one region should get the same class distribution: mean over labeled pixels
    of ||p_pixel - p_region||^2, where p_region is the region's mean probability. Regions
    are the ground-truth connected components of each class (gt_regions; default), or SLIC
    superpixels of the image (slic_segments; color-based, can cross true borders).
    Unlike cross-entropy, it charges extra for a true region predicted as a patchwork of
    classes, and nothing extra for one predicted coherently.

presence_bce
    Tile-level class presence, as the presence rule: class k is present when it covers
    >= 1% of the labeled pixels. The model's soft presence q_k is the mean probability of
    its 1% most confident pixels for k (so "present" means "at least 1% of the tile says k");
    loss = binary cross-entropy against the true class set.

presence_prior
    Optional: expected rarity of the predicted class set, sum_S P_q(S) * -log P_fit(S),
    with independent Bernoulli q_k and the fit-split class-set prior used by the presence
    rule (configs/tables/bcss_presence_prior.json, wcsp order).
"""
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

# BCSS order: tumor, stroma, inflammatory, necrosis, other -> wcsp order index
BCSS_TO_WCSP = (0, 2, 1, 3, 4)


def slic_segments(images, n_segments=256, compactness=10.0):
    """images: (B, 3, H, W) float 0-255 -> (B, H, W) long segment ids (CPU SLIC, Lab)."""
    from skimage.segmentation import slic

    arrays = images.detach().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    segs = [slic(a, n_segments=n_segments, compactness=compactness, start_label=0) for a in arrays]
    return torch.from_numpy(np.stack(segs)).long().to(images.device)


def gt_regions(target, valid, num_classes=5):
    """(B, H, W) long ids: 8-connected components of each ground-truth class (ignored px -> 0)."""
    from scipy import ndimage

    t = target.detach().cpu().numpy()
    ok = valid.detach().cpu().numpy()
    out = np.zeros(t.shape, dtype=np.int64)
    for i in range(len(t)):
        nxt = 1
        for c in range(num_classes):
            comp, n = ndimage.label((t[i] == c) & ok[i], structure=np.ones((3, 3)))
            out[i][comp > 0] = comp[comp > 0] + nxt - 1
            nxt += n
    return torch.from_numpy(out).to(target.device)


def cluster_consistency(probs, segments, valid):
    """probs (B, K, H, W), segments (B, H, W), valid (B, H, W) bool."""
    b, k = probs.shape[:2]
    per = int(segments.max()) + 1
    ids = (segments + per * torch.arange(b, device=probs.device).view(b, 1, 1))[valid]
    p = probs.permute(0, 2, 3, 1)[valid]                        # (N, K)
    sums = torch.zeros(b * per, k, device=probs.device, dtype=probs.dtype).index_add_(0, ids, p)
    counts = torch.zeros(b * per, device=probs.device, dtype=probs.dtype).index_add_(
        0, ids, torch.ones_like(ids, dtype=probs.dtype))
    mean = sums / counts.clamp_min(1).unsqueeze(1)
    return ((p - mean[ids]) ** 2).sum(1).mean()


def soft_presence(probs, valid, share=0.01):
    """(B, K) mean probability of the top `share` of labeled pixels, per class."""
    b, k = probs.shape[:2]
    flat = (probs * valid.unsqueeze(1)).flatten(2)              # unlabeled pixels -> 0
    out = []
    for i in range(b):
        n = max(1, math.ceil(share * int(valid[i].sum())))
        out.append(flat[i].topk(n, dim=1).values.mean(1))
    return torch.stack(out)


def true_presence(target, valid, num_classes, share=0.01):
    counts = torch.stack([((target == c) & valid).flatten(1).sum(1) for c in range(num_classes)], 1)
    return (counts.float() >= share * valid.flatten(1).sum(1, keepdim=True).float()).float()


def presence_bce(probs, target, valid, share=0.01):
    keep = valid.flatten(1).any(1)
    q = soft_presence(probs[keep], valid[keep], share).clamp(1e-6, 1 - 1e-6)
    y = true_presence(target[keep], valid[keep], probs.shape[1], share)
    return F.binary_cross_entropy(q, y)


class PresencePrior:
    """-log P(set) over the 31 class sets, indexed by BCSS-order bitmask."""

    def __init__(self, path, num_classes=5):
        cost_wcsp = {int(m): v for m, v in json.loads(open(path).read())["cost"].items()}
        floor = max(cost_wcsp.values())
        cost = np.full(2 ** num_classes, floor)
        for mask in range(1, 2 ** num_classes):
            wmask = sum(1 << BCSS_TO_WCSP[c] for c in range(num_classes) if mask >> c & 1)
            cost[mask] = cost_wcsp.get(wmask, floor)
        cost[0] = floor
        self.cost = torch.tensor(cost, dtype=torch.float32)
        bits = np.array([[m >> c & 1 for c in range(num_classes)] for m in range(2 ** num_classes)])
        self.bits = torch.tensor(bits, dtype=torch.float32)       # (2^K, K)

    def expected_cost(self, q):
        """q (B, K) presence probabilities -> mean expected -log P(set)."""
        bits, cost = self.bits.to(q.device), self.cost.to(q.device)
        logp = bits @ q.clamp(1e-6, 1).log().T + (1 - bits) @ (1 - q).clamp(1e-6, 1).log().T   # (2^K, B)
        return (logp.exp() * cost.unsqueeze(1)).sum(0).mean()

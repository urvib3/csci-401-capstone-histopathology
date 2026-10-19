# GridProblem: one pixel-labeling WCSP, stored as a few dense arrays.
#
# Every pixel is a variable over the wcsp labels. The cost of a labeling x is
#
#   constant
#   + sum over active p of unary[p, x_p]
#   + sum over offsets o, over active pairs (p, q = p + offsets[o]) of
#         weights[o] * (table[x_p, x_q] + change[o, p] * (x_p != x_q))
#
# "Active" pixels are the variables being optimized. Inactive pixels are
# fixed (or missing): their costs have been folded into the unaries of their
# active neighbors and into `constant` (see wcsp.solve.condition).

from dataclasses import dataclass

import numpy as np

from ..config import ProblemConfig
from .neighborhood import distance_weights, forward_offsets
from .pairwise import color_features, contrast_term, instance_term
from .tables import check_table, load_table
from .unary import unary_costs


@dataclass(frozen=True)
class GridProblem:
    unary: np.ndarray                 # (H, W, K) float
    offsets: np.ndarray               # (n, 2) forward offsets
    weights: np.ndarray               # (n,) lam / d^eta per offset
    table: np.ndarray                 # (K, K) symmetric b table
    change: np.ndarray | None = None  # (n, H, W) extra cost when labels differ
    active: np.ndarray | None = None  # (H, W) bool; None = all active
    constant: float = 0.0

    def __post_init__(self):
        height, width, k = self.unary.shape
        n = len(self.offsets)
        if self.offsets.shape != (n, 2) or self.weights.shape != (n,):
            raise ValueError("offsets must be (n, 2) and weights (n,).")
        check_table(self.table, num_labels=k)
        if self.change is not None and self.change.shape != (n, height, width):
            raise ValueError(f"change must be ({n}, {height}, {width}).")
        if self.active is not None and self.active.shape != (height, width):
            raise ValueError(f"active must be ({height}, {width}).")

    @property
    def shape(self):
        return self.unary.shape[:2]

    @property
    def num_labels(self):
        return self.unary.shape[2]

    @property
    def active_mask(self):
        if self.active is None:
            return np.ones(self.shape, dtype=bool)
        return self.active

    @property
    def num_variables(self):
        return int(self.active_mask.sum())


def build_problem(probs, config=ProblemConfig(), image=None, instances=None,
                  informative=None):
    """
    Build the WCSP for one image from classifier probabilities.

    probs:
        (K, H, W) class probabilities in wcsp label order.
    image:
        (H, W, 3) uint8, needed when config.contrast_beta > 0.
    instances:
        (H, W) nucleus ids, needed when config.instance_gamma > 0.
    informative:
        (H, W) bool; pixels where it is False get zero unary cost.
    """
    probs = np.asarray(probs)
    shape = probs.shape[1:]
    offsets = forward_offsets(config.types)
    weights = config.lam * distance_weights(offsets, config.eta, config.normalize)

    change = None
    if config.contrast_beta > 0:
        if image is None:
            raise ValueError("contrast_beta > 0 needs the image.")
        features = color_features(image, config.color_space)
        change = contrast_term(features, offsets, config.contrast_beta, config.contrast_sigma)
    if config.instance_gamma > 0:
        if instances is None:
            raise ValueError("instance_gamma > 0 needs the instance map.")
        term = instance_term(instances, offsets, config.instance_gamma)
        change = term if change is None else change + term
    if change is not None and change.shape[1:] != shape:
        raise ValueError("image / instances must match the probability map size.")

    return GridProblem(
        unary=config.alpha * unary_costs(probs, config.temperature, informative),
        offsets=offsets,
        weights=weights,
        table=load_table(config.table),
        change=change,
    )

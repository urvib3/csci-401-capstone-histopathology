# The five cell labels every part of wcsp/ works in.
#
# The order follows the cost table in the project notes, which is also the
# PanNuke channel order. Datasets and classifiers that use another order are
# converted once, on the way in, with remap_labels / reorder_probs.

import numpy as np

LABELS = ("neoplastic", "inflammatory", "connective", "dead", "healthy")
NUM_LABELS = len(LABELS)
NEOPLASTIC, INFLAMMATORY, CONNECTIVE, DEAD, HEALTHY = range(NUM_LABELS)

# Pixels with no usable ground truth. Same value as bcss.IGNORE_INDEX.
IGNORE = 255

# RGB colours for overlays, in LABELS order.
COLORS = np.array([
    [220, 40, 40],    # neoplastic: red
    [40, 160, 40],    # inflammatory: green
    [60, 90, 220],    # connective: blue
    [240, 200, 30],   # dead: yellow
    [230, 120, 200],  # healthy: pink
], dtype=np.uint8)

# Source index -> wcsp index. BCSS coarse order is tumor, stroma,
# inflammatory, necrosis, other (bcss.coarse_lut).
BCSS_TO_WCSP = np.array([NEOPLASTIC, CONNECTIVE, INFLAMMATORY, DEAD, HEALTHY])
# PanNuke channels 0-4 are neoplastic, inflammatory, connective, dead,
# non-neoplastic epithelial: already in wcsp order.
PANNUKE_TO_WCSP = np.arange(NUM_LABELS)


def _check_mapping(mapping):
    mapping = np.asarray(mapping)
    if sorted(mapping.tolist()) != list(range(NUM_LABELS)):
        raise ValueError(f"mapping must be a permutation of 0..{NUM_LABELS - 1}.")
    return mapping


def remap_labels(mask, mapping):
    """
    Convert a label mask from a source order to wcsp order.

    mapping:
        mapping[source_label] = wcsp_label. IGNORE is kept; any other value
        outside the mapping also becomes IGNORE.
    """
    mapping = _check_mapping(mapping)
    lut = np.full(256, IGNORE, dtype=np.uint8)
    lut[:NUM_LABELS] = mapping
    return lut[np.asarray(mask, dtype=np.uint8)]


def reorder_probs(probs, mapping, axis=0):
    """Move the class axis of probabilities (or costs) into wcsp order."""
    mapping = _check_mapping(mapping)
    probs = np.moveaxis(np.asarray(probs), axis, 0)
    out = np.empty_like(probs)
    out[mapping] = probs
    return np.moveaxis(out, 0, axis)


def colorize(labels):
    """(H, W) labels -> (H, W, 3) uint8 image; IGNORE is black."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    valid = labels < NUM_LABELS
    out[valid] = COLORS[labels[valid]]
    return out

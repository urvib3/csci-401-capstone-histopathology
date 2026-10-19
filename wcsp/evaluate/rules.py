# How often a labeling breaks the biological rules of the notes.
#
# Counts neighboring pixel pairs (all five neighbor types) with the label
# combinations each rule penalizes. Comparing ground truth, classifier and
# WCSP shows whether the rules hold in real tissue and what the WCSP changes.

import numpy as np

from ..labels import CONNECTIVE, DEAD, HEALTHY, INFLAMMATORY, NEOPLASTIC, NUM_LABELS
from ..model.neighborhood import forward_offsets, pair_slices

RULE_PAIRS = {
    "necrosis_isolation": (DEAD, HEALTHY),
    "immune_infiltration": (INFLAMMATORY, HEALTHY),
    "stromal_boundary": (NEOPLASTIC, HEALTHY),
    "stroma_dead": (CONNECTIVE, DEAD),
}


def pair_counts(labels, valid=None, offsets=None):
    """
    (K, K) symmetric counts of neighboring label pairs.

    valid:
        Optional (H, W) bool; pairs touching an invalid pixel are skipped.
        Labels >= K (e.g. IGNORE) are always skipped.
    """
    labels = np.asarray(labels, dtype=np.int64)
    offsets = forward_offsets() if offsets is None else offsets
    ok = labels < NUM_LABELS
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    counts = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    for offset in offsets:
        p, q = pair_slices(labels.shape, offset)
        both = ok[p] & ok[q]
        np.add.at(counts, (labels[p][both], labels[q][both]), 1)
    return counts + counts.T - np.diag(np.diag(counts))


def rule_report(counts):
    """Per-rule violating pairs per 10,000 neighboring pairs, plus unlike-pair rate."""
    counts = np.asarray(counts)
    total = np.triu(counts).sum()
    scale = 1e4 / total if total else float("nan")
    report = {name: float(counts[a, b] * scale) for name, (a, b) in RULE_PAIRS.items()}
    report["unlike_pairs"] = float((total - np.trace(counts)) * scale)
    return report

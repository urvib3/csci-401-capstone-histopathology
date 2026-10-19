# Label-compatibility tables: the b_ij terms of the notes.
#
# b[i, j] is the cost of label i at pixel p next to label j at pixel q,
# before division by the distance term. Rows and columns are in wcsp label
# order (wcsp.labels.LABELS).

import csv
from pathlib import Path

import numpy as np

from ..labels import LABELS, NUM_LABELS

# The hand-written table from the project notes.
#   0  homophily: same label is free
#   10 necrosis isolation: dead next to healthy
#   9  stromal boundary: tumor next to healthy
#   8  immune infiltration: immune next to healthy
#   5  connective next to dead is mildly penalized
#   1, 2 expected biology
HANDWRITTEN = np.array([
    #  neo  inf  con  dead  healthy
    [0, 2, 1, 1, 9],    # neoplastic
    [2, 0, 2, 1, 8],    # inflammatory
    [1, 2, 0, 5, 1],    # connective
    [1, 1, 5, 0, 10],   # dead
    [9, 8, 1, 10, 0],   # healthy
], dtype=np.float64)

POTTS = 1.0 - np.eye(NUM_LABELS)

# For BCSS, whose fifth class is "other" (fat, vessels, blood, normal ducts,
# ...) rather than healthy epithelium: keep the handwritten rules among the
# first four labels but make "other" a neutral neighbor (cost 1 to every
# other label), so the healthy-tissue rules are not applied to it.
OTHER_NEUTRAL = HANDWRITTEN.copy()
OTHER_NEUTRAL[4, :4] = OTHER_NEUTRAL[:4, 4] = 1.0


def check_table(table, num_labels=NUM_LABELS):
    """Validate a compatibility table and return it as float64."""
    table = np.asarray(table, dtype=np.float64)
    if table.shape != (num_labels, num_labels):
        raise ValueError(f"table must be {num_labels}x{num_labels}, got {table.shape}.")
    if (table < 0).any():
        raise ValueError("table costs must be non-negative.")
    if not np.allclose(table, table.T):
        raise ValueError("table must be symmetric (each pixel pair is counted once).")
    return table


def load_table(spec):
    """
    A table by name ("handwritten", "other_neutral", "potts") or from a CSV file.

    CSV format: a header row of label names, then one row per label whose
    first cell is the label name. Labels must be in wcsp order.
    """
    if isinstance(spec, np.ndarray):
        return check_table(spec)
    named = {"handwritten": HANDWRITTEN, "other_neutral": OTHER_NEUTRAL, "potts": POTTS}
    if spec in named:
        return named[spec].copy()

    with Path(spec).open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0][1:], rows[1:]
    if tuple(header) != LABELS or tuple(r[0] for r in body) != LABELS:
        raise ValueError(f"{spec}: rows and columns must be {LABELS}.")
    return check_table([[float(v) for v in r[1:]] for r in body])


def save_table(table, path):
    table = check_table(table)
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *LABELS])
        for name, row in zip(LABELS, table):
            writer.writerow([name, *(f"{v:g}" for v in row)])

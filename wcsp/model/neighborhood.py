# Which pixels interact, and how strongly.
#
# A pixel p interacts with every pixel q within distance 2*sqrt(2), split
# into the five neighbor types of the project notes:
#
#   type 1: distance 1        (0, +-1), (+-1, 0)        4 neighbors
#   type 2: distance sqrt(2)  (+-1, +-1)                4
#   type 3: distance 2        (0, +-2), (+-2, 0)        4
#   type 4: distance sqrt(5)  (+-1, +-2), (+-2, +-1)    8
#   type 5: distance 2sqrt(2) (+-2, +-2)                4
#
# Each unordered pair {p, q} is counted once, so problems store only the 12
# "forward" offsets (dy > 0, or dy == 0 and dx > 0); the other 12 are their
# negatives. Pixels near the image border simply have fewer neighbors.

import numpy as np

# Squared distance of each neighbor type.
TYPE_DISTANCE_SQ = {1: 1, 2: 2, 3: 4, 4: 5, 5: 8}
ALL_TYPES = tuple(TYPE_DISTANCE_SQ)


def neighbor_type(offset):
    """Neighbor type (1-5) of an offset (dy, dx)."""
    dy, dx = offset
    d2 = dy * dy + dx * dx
    for kind, value in TYPE_DISTANCE_SQ.items():
        if value == d2:
            return kind
    raise ValueError(f"offset {offset} is not a neighbor.")


def forward_offsets(types=ALL_TYPES):
    """
    The forward offsets for the given neighbor types.

    returns:
        int array of shape (n, 2), rows (dy, dx), sorted by distance.
    """
    wanted = {TYPE_DISTANCE_SQ[t] for t in types}
    offsets = [
        (dy, dx)
        for dy in range(0, 3)
        for dx in range(-2, 3)
        if (dy > 0 or dx > 0) and dy * dy + dx * dx in wanted
    ]
    offsets.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    return np.array(offsets, dtype=np.int64).reshape(-1, 2)


def distance_weights(offsets, eta, normalize=False):
    """
    1 / d^eta for each offset, the d_ij term of the notes.

    normalize:
        Divide by the total weight of a full (both-direction) neighborhood,
        so that changing eta or the neighbor types does not change the
        overall strength of the pairwise terms.
    """
    if eta < 0:
        raise ValueError("eta must be >= 0.")
    offsets = np.asarray(offsets, dtype=np.float64)
    distance = np.sqrt((offsets ** 2).sum(axis=1))
    weights = distance ** -eta
    if normalize:
        weights = weights / (2 * weights.sum())
    return weights


def pair_slices(shape, offset):
    """
    Slices selecting every pixel pair (p, q = p + offset) inside the image.

    returns:
        (p, q): tuples of slices such that array[p] and array[q] line up
        pair by pair.
    """
    height, width = shape
    dy, dx = (int(v) for v in offset)

    def axis(size, d):
        return slice(max(0, -d), size - max(0, d)), slice(max(0, d), size + min(0, d))

    p_rows, q_rows = axis(height, dy)
    p_cols, q_cols = axis(width, dx)
    return (p_rows, p_cols), (q_rows, q_cols)

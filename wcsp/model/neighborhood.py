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


def sample_long_edges(shape, distance, probability, seed=0, max_edges=2_000_000):
    """Independent Bernoulli(p) edges over unordered pairs beyond distance.

    Sample the upper triangle via geometric skips, without enumerating N²
    candidates. Local neighbors (distance <= 2 sqrt(2)) are never duplicated.
    The explicit memory guard rejects dense requests; it does not truncate.
    """
    if not np.isfinite(distance) or distance < np.sqrt(8):
        raise ValueError("long_distance must be finite and >= sqrt(8).")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("long_probability must be in [0, 1].")
    if max_edges < 1:
        raise ValueError("long_max_edges must be positive.")
    n = int(np.prod(shape))
    total = n * (n - 1) // 2
    if probability == 0 or total == 0:
        return np.empty((0, 2), dtype=np.int64)
    # Conservative guard on candidates, before the distance filter.
    if total * probability > max_edges:
        raise ValueError("expected long-edge candidates exceed long_max_edges; reduce p.")
    rng = np.random.default_rng(seed)
    chunks, cursor, count = [], -1, 0
    while cursor < total - 1:
        batch = min(65536, max(1, int(np.ceil((total-cursor)*probability)) + 8))
        gaps = (np.ones(batch, dtype=np.int64) if probability == 1 else
                rng.geometric(probability, size=batch))
        # Extremely small p may overflow int64 in geometric sampling.
        if (gaps <= 0).any():
            raise ValueError("long_probability is too small for geometric sampling.")
        ranks = cursor + np.cumsum(gaps)
        ranks = ranks[(ranks >= 0) & (ranks < total)]
        if not len(ranks):
            break
        count += len(ranks)
        if count > max_edges:
            raise ValueError("sampled candidates exceed long_max_edges; reduce p.")
        first = np.floor((2*n - 1 - np.sqrt((2*n-1)**2 - 8.0*ranks))/2).astype(np.int64)
        second = ranks - first * (2*n-first-1)//2 + first + 1
        width = shape[1]
        d2 = (first//width-second//width)**2 + (first%width-second%width)**2
        chunks.append(np.stack([first[d2 > distance**2], second[d2 > distance**2]], axis=1))
        if len(ranks) < len(gaps):
            break
        cursor = int(ranks[-1])
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int64)

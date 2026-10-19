# Image-dependent pairwise terms: the f_ij terms of the notes.
#
# Both terms are added only when the two pixels take different labels, so
# same-label pairs stay free (homophily). They are stored per forward offset
# as an (n_offsets, H, W) array: entry [o, y, x] belongs to the pair
# (p = (y, x), q = p + offsets[o]); entries whose q is outside the image are 0.

import numpy as np

from .neighborhood import pair_slices


def color_features(image, space="rgb"):
    """
    Per-pixel colour features scaled to roughly 0-1.

    space:
        "rgb", or "hed" for hematoxylin / eosin / DAB optical density
        (needs scikit-image).
    """
    image = np.asarray(image)
    if space == "rgb":
        return image.astype(np.float64) / 255.0
    if space == "hed":
        from skimage.color import rgb2hed

        hed = rgb2hed(image)
        low, high = hed.reshape(-1, 3).min(0), hed.reshape(-1, 3).max(0)
        return (hed - low) / np.maximum(high - low, 1e-9)
    raise ValueError(f"unknown colour space {space!r}.")


def contrast_term(features, offsets, beta, sigma):
    """
    beta * exp(-||z_p - z_q||^2 / (2 sigma^2)).

    Changing label between similar-looking pixels costs up to beta extra;
    changing across a strong colour edge costs almost nothing.

    features:
        (H, W, C) per-pixel features, e.g. color_features(image).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 2:
        features = features[..., None]
    out = np.zeros((len(offsets),) + features.shape[:2])
    for o, offset in enumerate(offsets):
        p, q = pair_slices(features.shape[:2], offset)
        d2 = ((features[p] - features[q]) ** 2).sum(axis=-1)
        out[o][p] = beta * np.exp(-d2 / (2 * sigma ** 2))
    return out


def instance_term(instances, offsets, gamma):
    """
    gamma for pixel pairs inside the same nucleus, 0 otherwise.

    instances:
        (H, W) int nucleus ids, 0 = background. A large gamma makes every
        detected nucleus take one label.
    """
    instances = np.asarray(instances)
    out = np.zeros((len(offsets),) + instances.shape)
    for o, offset in enumerate(offsets):
        p, q = pair_slices(instances.shape, offset)
        same = (instances[p] == instances[q]) & (instances[p] != 0)
        out[o][p] = gamma * same
    return out

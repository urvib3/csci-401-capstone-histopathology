# BCSS label definitions and class groupings.
#
# The original BCSS masks (and the 512x512 Kaggle variant) use the 22 ground
# truth codes from PathologyDataScience/BCSS meta/gtruth_codes.tsv. Code 0
# (outside_roi) is a "don't care" region, not a tissue class, and should be
# ignored when scoring or fitting costs. Code 7 (exclude) is likewise
# unusable annotation.
#
# The 224x224 Kaggle variant has already been merged down to 3 labels by the
# uploader; see BCSS224_LABELS.

import numpy as np

IGNORE_INDEX = 255

BCSS_CODES = {
    0: "outside_roi",
    1: "tumor",
    2: "stroma",
    3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris",
    5: "glandular_secretions",
    6: "blood",
    7: "exclude",
    8: "metaplasia_NOS",
    9: "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined",
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other",
}

# Codes that carry no usable label.
BCSS_DONT_CARE = (0, 7)

# Coarse grouping used by the BCSS paper baseline (Amgad et al. 2019):
# tumor / stroma / inflammatory / necrosis / other. The 22-class problem is
# heavily long-tailed, so this is the usual starting point.
COARSE_GROUPS = {
    "tumor": (1, 19, 20),
    "stroma": (2,),
    "inflammatory": (3, 10, 11),
    "necrosis": (4,),
    "other": (5, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 21),
}

# The 224x224 Kaggle variant's masks contain values {0, 1, 2}. Determined by
# overlaying 224 tiles on the 512 tiles of the same ROI: 1 = tumor codes
# (1, 19, 20), 2 = stroma (2), 0 = every other code. Note that 0 also
# absorbs outside_roi and exclude, so the 224 masks have no don't-care
# label.
BCSS224_LABELS = {0: "other", 1: "tumor", 2: "stroma"}


def make_lut(groups, dont_care=BCSS_DONT_CARE, num_codes=256):
    """
    Build a lookup table mapping raw mask codes to new class indices.

    groups:
        An ordered mapping {class name: tuple of raw codes}. The i-th group
        becomes class index i.
    dont_care:
        Raw codes mapped to IGNORE_INDEX.

    returns:
        lut: uint8 array of length num_codes, used as lut[mask]. Codes not
             listed in groups or dont_care are also mapped to IGNORE_INDEX.
        names: list of class names, indexed by new class index.
    """
    lut = np.full(num_codes, IGNORE_INDEX, dtype=np.uint8)
    names = []
    for index, (name, codes) in enumerate(groups.items()):
        for code in codes:
            lut[code] = index
        names.append(name)
    for code in dont_care:
        lut[code] = IGNORE_INDEX
    return lut, names


def coarse_lut():
    """LUT from the 22 BCSS codes to the 5 coarse groups."""
    return make_lut(COARSE_GROUPS)


def full_lut():
    """LUT keeping the 20 real BCSS tissue classes, with 0 and 7 ignored."""
    groups = {
        name: (code,)
        for code, name in BCSS_CODES.items()
        if code not in BCSS_DONT_CARE
    }
    return make_lut(groups)

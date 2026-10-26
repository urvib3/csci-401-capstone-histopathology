"""Data access and WCSP-oriented primitives for the BCSS histopathology dataset."""

from .classes import (
    BCSS_CODES,
    COARSE_GROUPS,
    IGNORE_INDEX,
    coarse_lut,
    full_lut,
    make_lut,
)
from .dataset import BCSSDataset, Sample, TileInfo, default_root
from .grid import CellGrid, to_integer_costs, unary_costs_from_probabilities

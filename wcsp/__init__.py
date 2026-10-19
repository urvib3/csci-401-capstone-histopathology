"""Pixel-level WCSP labeling of histopathology images (PanNuke and BCSS)."""

from .config import ProblemConfig
from .labels import IGNORE, LABELS, NUM_LABELS, remap_labels, reorder_probs
from .model import GridProblem, build_problem
from .solve import (
    ArgmaxSolver,
    BlockCoordinateDescent,
    BlockLayout,
    ICMSolver,
    SolveResult,
    condition,
    energy,
)

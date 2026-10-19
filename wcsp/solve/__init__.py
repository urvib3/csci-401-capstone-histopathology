"""Solvers for GridProblem: energy, conditioning, baselines and the block scheme."""

from .argmax import ArgmaxSolver
from .base import SolveResult, Solver
from .blocks import BlockCoordinateDescent, BlockLayout
from .condition import condition
from .energy import energy, local_costs, neighbor_costs
from .icm import ICMSolver

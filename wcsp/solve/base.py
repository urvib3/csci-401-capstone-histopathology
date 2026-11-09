# The interface every solver implements.

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class SolveResult:
    labels: np.ndarray                 # (H, W) labels; inactive pixels keep init
    energy: float
    seconds: float
    lower_bound: float | None = None   # when the solver proves one
    trace: list = field(default_factory=list)  # per-step records, e.g. energy per round
    info: dict = field(default_factory=dict)   # per-sample extras saved as CSV columns


class Solver(Protocol):
    def solve(self, problem, init) -> SolveResult:
        """Minimize problem's energy over its active pixels, starting from init."""
        ...

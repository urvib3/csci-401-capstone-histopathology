# Solvers from config dictionaries, e.g.
#   {name: blocks, block: 100, max_rounds: 10, inner: {name: toulbar2, time_limit: 20}}

from .argmax import ArgmaxSolver
from .blocks import BlockCoordinateDescent, BlockLayout
from .chain import ChainSolver
from .icm import ICMSolver


def get_solver(spec):
    spec = dict(spec)
    name = spec.pop("name")
    if name == "argmax":
        return ArgmaxSolver()
    if name == "icm":
        return ICMSolver(**spec)
    if name == "toulbar2":
        from .toulbar2 import Toulbar2Solver

        return Toulbar2Solver(**spec)
    if name == "blocks":
        inner = get_solver(spec.pop("inner", {"name": "icm"}))
        block = spec.pop("block", 100)
        whole = spec.pop("whole", False)
        tile = spec.pop("tile", None)

        def layout_for(shape):
            if whole:
                return BlockLayout.whole(shape)
            if tile:
                return BlockLayout.tiles(shape, tile)
            return BlockLayout.cross(shape, block)

        return BlockCoordinateDescent(inner, layout=layout_for, **spec)
    if name == "chain":
        return ChainSolver([get_solver(stage) for stage in spec.pop("stages")])
    raise ValueError(f"unknown solver {name!r}.")

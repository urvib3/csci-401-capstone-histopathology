# Solvers from config dictionaries, e.g.
#   {name: blocks, block: 100, max_rounds: 10, inner: {name: toulbar2, time_limit: 20}}
#   {name: presence, gamma: 100, prior: <json>, inner: {name: blocks, ...}}
#   {name: hierarchy, levels: 3, mu: 1, pixel: {name: blocks, block: 100, max_rounds: 4}}
#   {name: clusters, n_segments: 256, features: both, refine: {name: blocks, ...}}
#   {name: min_region, min_area: {3: 1024}, inner: {name: blocks, ...}}

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
    if name == "hierarchy":
        from .hierarchy import HierarchySolver

        pixel = get_solver(spec.pop("pixel", {"name": "blocks", "block": 100, "max_rounds": 4}))
        return HierarchySolver(pixel, **spec)
    if name == "presence":
        from .presence import PresenceSolver

        inner = get_solver(spec.pop("inner", {"name": "blocks", "block": 100, "max_rounds": 4}))
        return PresenceSolver(inner, **spec)
    if name == "clusters":
        from .clusters import ClusterSolver

        refine = spec.pop("refine", None)
        return ClusterSolver(refine=get_solver(refine) if refine else None, **spec)
    if name == "min_region":
        from .regions import MinRegionSolver

        inner = get_solver(spec.pop("inner", {"name": "blocks", "block": 100, "max_rounds": 4}))
        return MinRegionSolver(inner, **spec)
    if name == "chain":
        return ChainSolver([get_solver(stage) for stage in spec.pop("stages")])
    raise ValueError(f"unknown solver {name!r}.")

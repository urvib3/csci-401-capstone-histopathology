"""Dataset adapters: every dataset yields wcsp.data.Sample in wcsp label order."""

from .base import Sample, parse_shard, select


def load_dataset(name, split, root=None, **options):
    """load_dataset("bcss", "dev") or load_dataset("pannuke", "test", fold=2)."""
    if name == "bcss":
        from .bcss import BCSS

        return BCSS(split, root)
    if name == "pannuke":
        from .pannuke import PanNuke

        return PanNuke(split, root, **options)
    raise ValueError(f"unknown dataset {name!r}.")

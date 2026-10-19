# Settings for building a WCSP from classifier probabilities, and the
# experiment files (configs/*.yaml) that tie a dataset, a classifier, the
# problem settings and a solver together.

import copy
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProblemConfig:
    """
    Knobs of the energy

        E(x) = alpha * sum_p U_p(x_p)
             + lam * sum_{p,q} [b(x_p, x_q) + f_pq * (x_p != x_q)] / d_pq^eta
    """

    alpha: float = 1.0               # weight of classifier fidelity
    lam: float = 1.0                 # weight of the pairwise rules
    eta: float = 1.0                 # distance exponent, d_pq^eta
    types: tuple = (1, 2, 3, 4, 5)   # neighbor types used (wcsp.model.neighborhood)
    normalize: bool = True           # keep total pairwise weight fixed across eta / types
    table: str = "handwritten"       # "handwritten", "potts" or a CSV path
    temperature: float = 1.0         # classifier calibration
    contrast_beta: float = 0.0       # 0 disables the colour-contrast f term
    contrast_sigma: float = 0.1
    color_space: str = "rgb"
    instance_gamma: float = 0.0      # 0 disables the same-nucleus f term (PanNuke)

    @classmethod
    def from_dict(cls, values):
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown problem settings: {sorted(unknown)}")
        values = dict(values)
        if "types" in values:
            values["types"] = tuple(values["types"])
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        values["types"] = list(self.types)
        return values


@dataclass
class ExperimentConfig:
    """
    One configs/*.yaml file:

        name: bcss_unet
        dataset:    {name: bcss, root: ..., split: dev}
        classifier: {name: tia_unet, temperature: auto}
        problem:    {...ProblemConfig fields...}
        solver:     {name: blocks, inner: {name: toulbar2, time_limit: 20}}
        paths:      {cache: ..., runs: ...}

    Paths may use environment variables, e.g. $WCSP_OUT.
    """

    name: str
    dataset: dict
    classifier: dict
    problem: ProblemConfig
    solver: dict
    paths: dict = field(default_factory=dict)

    @property
    def cache_root(self):
        return Path(os.path.expandvars(self.paths.get("cache", "$WCSP_OUT/cache")))

    @property
    def runs_root(self):
        return Path(os.path.expandvars(self.paths.get("runs", "$WCSP_OUT/runs")))

    def to_dict(self):
        return {
            "name": self.name,
            "dataset": self.dataset,
            "classifier": self.classifier,
            "problem": self.problem.to_dict(),
            "solver": self.solver,
            "paths": self.paths,
        }


def apply_overrides(values, overrides):
    """overrides: ["problem.lam=2", "solver.inner.time_limit=5"]; values parsed as YAML."""
    values = copy.deepcopy(values)
    for item in overrides or ():
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value.")
        node = values
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(text)
    return values


def load_config(path, overrides=()):
    values = yaml.safe_load(Path(path).read_text())
    values = apply_overrides(values, overrides)
    if "root" in values.get("dataset", {}):
        values["dataset"]["root"] = os.path.expandvars(values["dataset"]["root"])
    return ExperimentConfig(
        name=values.get("name", Path(path).stem),
        dataset=values["dataset"],
        classifier=values["classifier"],
        problem=ProblemConfig.from_dict(values.get("problem", {})),
        solver=values.get("solver", {"name": "argmax"}),
        paths=values.get("paths", {}),
    )

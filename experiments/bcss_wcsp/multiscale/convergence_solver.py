"""Instrumented local multiscale descent, isolated from frozen historical solvers.

Finite limits reproduce the original update order. None removes a limit.
Convergence is audited against all single-variable moves, not global optimality.
"""
from dataclasses import dataclass, field, replace
import time

import numpy as np

from wcsp.solve.base import SolveResult
from wcsp.solve.blocks import BlockLayout
from wcsp.solve.condition import condition
from wcsp.solve.energy import energy, local_costs
from wcsp.solve.hierarchy import HierarchySolver, one_hot, pool
from wcsp.solve.instrumentation import emit, solver_scope


@dataclass
class MeasuredResult(SolveResult):
    levels: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def check_limit(value):
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError('Iteration limits must be positive integers or None')


class MeasuredICM:
    def __init__(self, max_sweeps=None):
        check_limit(max_sweeps)
        self.max_sweeps = max_sweeps

    def solve(self, problem, init):
        if (np.abs(problem.offsets).max(initial=0) >= 3
                or (problem.long_edges is not None and len(problem.long_edges))):
            raise ValueError('This experiment supports the original local neighborhood only')
        start = time.perf_counter()
        labels = np.array(init, dtype=np.int64)
        active = problem.active_mask
        rows, cols = np.indices(problem.shape)
        colors = (rows % 3) * 3 + cols % 3
        batches = [active & (colors == c) for c in np.unique(colors[active])]
        trace, updates = [energy(problem, labels)], []
        while self.max_sweeps is None or len(updates) < self.max_sweeps:
            changed = 0
            for batch in batches:
                costs = local_costs(problem, labels)
                best = costs.argmin(axis=-1)
                current = np.take_along_axis(costs, labels[..., None], -1)[..., 0]
                better = batch & (costs.min(axis=-1) < current - 1e-12)
                labels[better] = best[better]
                changed += int(better.sum())
            trace.append(energy(problem, labels))
            updates.append(changed)
            emit('icm_sweep', sweep=len(updates), changed_pixels=changed,
                 energy=trace[-1], variables=problem.num_variables,
                 elapsed_seconds=time.perf_counter() - start)
            if not changed:
                break
        stop = 'no_changes' if updates[-1] == 0 else 'sweep_limit'
        stats = dict(solver='icm', sweeps=len(updates), max_sweeps=self.max_sweeps,
                     pixel_updates_per_sweep=updates, energies=trace,
                     stop_reason=stop, converged=stop == 'no_changes')
        emit('icm_end', sweeps=len(updates), stop_reason=stop)
        return MeasuredResult(labels, trace[-1], time.perf_counter() - start,
                              trace=trace, stats=stats)


class MeasuredBlocks:
    def __init__(self, inner, block=100, max_rounds=None):
        check_limit(max_rounds)
        self.inner, self.block, self.max_rounds = inner, block, max_rounds

    def solve(self, problem, init):
        if problem.long_edges is not None and len(problem.long_edges):
            raise ValueError('Distant edges are outside this experiment')
        start = time.perf_counter()
        labels = np.array(init, dtype=np.int64)
        layout = BlockLayout.cross(problem.shape, self.block)
        active, current = problem.active_mask, energy(problem, labels)
        trace = [dict(round=0, phase=None, energy=current, seconds=0.)]
        details, skipped, rounds = [], 0, 0
        stop = 'round_limit'
        while self.max_rounds is None or rounds < self.max_rounds:
            rounds += 1
            before_round = current
            for phase, regions in enumerate(layout.phases):
                jobs = []
                # Preserve the historical conditioning and update order exactly.
                for region in regions:
                    region = region & active
                    if region.any():
                        sub, window = condition(problem, labels, region)
                        before = labels[window].copy()
                        bound = sub.constant + sub.unary[sub.active_mask].min(-1).sum()
                        if energy(sub, before) > bound + 1e-9:
                            jobs.append((sub, window, before))
                        else:
                            skipped += 1
                results = []
                for sub, window, before in jobs:
                    bounds = [[int(s.start), int(s.stop)] for s in window]
                    with solver_scope(round=rounds, phase=phase, window=bounds):
                        emit('block_start', variables=sub.num_variables)
                        result = self.inner.solve(sub, before)
                        emit('block_end', energy=result.energy)
                    results.append(result)
                for (sub, window, before), result in zip(jobs, results):
                    accepted = result.energy < energy(sub, before) - 1e-9
                    if accepted:
                        labels[window] = np.where(sub.active, result.labels, labels[window])
                    details.append(dict(round=rounds, phase=phase,
                                        window=[[int(s.start), int(s.stop)] for s in window],
                                        accepted=bool(accepted), stats=result.stats))
                current = energy(problem, labels)
                trace.append(dict(round=rounds, phase=phase, energy=current,
                                  seconds=time.perf_counter() - start))
            emit('block_round_end', block_round=rounds, energy=current)
            if current >= before_round - 1e-9:
                stop = 'energy_plateau'
                break
        stats = dict(solver='blocks', rounds=rounds, max_rounds=self.max_rounds,
                     stop_reason=stop, blocks=details, skipped_at_bound=skipped)
        return MeasuredResult(labels, current, time.perf_counter() - start,
                              trace=trace, stats=stats)


class MeasuredHierarchy(HierarchySolver):
    def __init__(self, pixel, *, block_max_sweeps=None, max_rounds=None, **kwargs):
        check_limit(max_rounds)
        super().__init__(pixel, max_rounds=max_rounds, **kwargs)
        self.block_solver = MeasuredICM(block_max_sweeps)

    def audit(self, problem, blocks, labels, block_labels, tolerance=1e-9):
        checks = []
        for level in range(self.levels + 1):
            base = problem if level == 0 else blocks[level - 1]
            y = labels if level == 0 else block_labels[level - 1]
            sub = replace(base, unary=base.unary + self._link_costs(
                level, problem.num_labels, labels, block_labels))
            costs = local_costs(sub, y)
            gain = np.take_along_axis(costs, y[..., None], -1)[..., 0] - costs.min(-1)
            checks.append(dict(level=level, improving_variables=int((gain > tolerance).sum()),
                               max_gain=float(gain.max(initial=0))))
        return dict(tolerance=tolerance, levels=checks,
                    improving_variables=sum(c['improving_variables'] for c in checks),
                    max_gain=max(c['max_gain'] for c in checks))

    def solve(self, problem, init, on_round=None):
        start = time.perf_counter()
        k, blocks = problem.num_labels, self.block_problems(problem)
        labels, block_labels = np.array(init, dtype=np.int64), []
        for level, sub in enumerate(blocks, start=1):
            below = labels if level == 1 else block_labels[-1]
            factor = self._factor(level)
            children = pool(one_hot(below, k), factor)
            cost = sub.unary + self.mu * self._child_area(level) * (factor**2 - children)
            block_labels.append(cost.argmin(-1))
        current = self.total_energy(problem, blocks, labels, block_labels)
        trace = [dict(round=0, level=None, energy=current, seconds=0.)]
        stages, round_rows, rounds = [], [], 0
        stop = 'round_limit'
        while self.max_rounds is None or rounds < self.max_rounds:
            rounds += 1
            round_start, changed_levels = current, []
            for level in range(self.levels + 1):
                before = labels.copy() if level == 0 else block_labels[level - 1].copy()
                base = problem if level == 0 else blocks[level - 1]
                sub = replace(base, unary=base.unary + self._link_costs(level, k, labels, block_labels))
                with solver_scope(hierarchy_round=rounds, level=level):
                    emit('hierarchy_level_start')
                    result = (self.pixel if level == 0 else self.block_solver).solve(sub, before)
                    changed = int((before != result.labels).sum())
                    emit('hierarchy_level_end', changed_variables=changed)
                if level == 0:
                    labels = result.labels
                else:
                    block_labels[level - 1] = result.labels
                current = self.total_energy(problem, blocks, labels, block_labels)
                if current > trace[-1]['energy'] + 1e-6:
                    raise ValueError('Hierarchy energy increased')
                trace.append(dict(round=rounds, level=level, energy=current,
                                  seconds=time.perf_counter() - start,
                                  changed_variables=changed))
                stages.append(dict(round=rounds, level=level, stats=result.stats))
                changed_levels.append(changed)
            row = dict(round=rounds, energy=current, improvement=round_start - current,
                       changed_pixels=changed_levels[0], changed_blocks=changed_levels[1:],
                       seconds=time.perf_counter() - start)
            round_rows.append(row)
            emit('hierarchy_round_end', **row)
            if on_round is not None:
                on_round(rounds, labels, block_labels, trace, stages, row['seconds'])
            if current >= round_start - self.tol:
                stop = 'energy_plateau'
                break
        audit = self.audit(problem, blocks, labels, block_labels)
        if stop == 'energy_plateau' and audit['improving_variables']:
            raise ValueError(f'Plateau failed single-variable stationarity audit: {audit}')
        stats = dict(solver='hierarchy', rounds=rounds, max_rounds=self.max_rounds,
                     stop_reason=stop, converged=audit['improving_variables'] == 0,
                     audit=audit, stages=stages, round_history=round_rows)
        return MeasuredResult(labels, current, time.perf_counter() - start,
                              trace=trace, levels=block_labels, stats=stats)


def make_solver(spec):
    pixel = spec['pixel']
    inner = pixel['inner']
    if spec['name'] != 'hierarchy' or pixel['name'] != 'blocks' or inner['name'] != 'icm':
        raise ValueError('The expanded experiment uses the matched hierarchy / blocks / ICM pipeline')
    return MeasuredHierarchy(
        MeasuredBlocks(MeasuredICM(inner['max_sweeps']), pixel['block'], pixel['max_rounds']),
        levels=spec['levels'], size=spec['size'], mu=spec['mu'],
        block_alpha=spec['block_alpha'], block_lam=spec['block_lam'],
        block_types=spec['block_types'], max_rounds=spec['max_rounds'],
        block_max_sweeps=spec['block_max_sweeps'])

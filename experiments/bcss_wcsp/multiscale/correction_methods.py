"""BCSS-only correction experiments; frozen historical solvers are untouched."""
from dataclasses import replace
import time

import numpy as np
from scipy import ndimage
from scipy.special import logsumexp

from experiments.bcss_wcsp.multiscale.convergence_solver import MeasuredResult
from wcsp.solve.energy import energy
from wcsp.solve.hierarchy import one_hot, pool
from wcsp.solve.instrumentation import emit, solver_scope


def appearance_features(image):
    """HED stain, local stain means and local standard deviations (nine features)."""
    from skimage.color import rgb2hed
    stain = rgb2hed(image).astype(np.float32)
    mean = ndimage.gaussian_filter(stain, sigma=(4, 4, 0))
    variance = ndimage.gaussian_filter(stain * stain, sigma=(4, 4, 0)) - mean * mean
    return np.concatenate([stain, mean, np.sqrt(np.maximum(variance, 0))], axis=-1)


def appearance_logits(features, model):
    means, variances = np.asarray(model['means']), np.asarray(model['variances'])
    return -.5 * (((features[..., None, :] - means) ** 2 / variances).sum(-1)
                  + np.log(variances).sum(-1))


def region_appearance_unary(image, segments, model):
    scores = appearance_logits(appearance_features(image), model) / model['temperature']
    probs = np.exp(scores - logsumexp(scores, axis=-1, keepdims=True))
    _, ids = np.unique(segments, return_inverse=True)
    ids = ids.ravel()
    counts = np.bincount(ids)
    means = np.stack([np.bincount(ids, weights=probs[..., k].ravel()) / counts
                      for k in range(probs.shape[-1])], axis=-1)
    costs = -np.log(np.clip(means, 1e-6, 1))
    costs -= costs.min(-1, keepdims=True)
    return costs[ids].reshape(probs.shape).astype(np.float32)


def presence_problem(problem, prior, gamma):
    """Existing two-stage presence objective; gamma=0 is an exact identity."""
    from wcsp.solve.presence import best_set, set_costs
    if gamma == 0:
        return problem, dict(chosen=31, gamma=0)
    costs = set_costs(problem.unary, np.asarray(prior), gamma)
    chosen = best_set(costs)
    allowed = np.array([bool(chosen & (1 << k)) for k in range(problem.num_labels)])
    # A finite, documented exclusion penalty; its final satisfaction is checked.
    sub = replace(problem, unary=problem.unary + 1000 * ~allowed)
    return sub, dict(chosen=chosen, gamma=gamma, allowed=allowed.tolist(), forbid=1000)


def region_masks(labels, segments=None, minimum=64, maximum=32):
    """Bound proposal work independently of annotation or method outcomes."""
    if segments is None:
        ids = np.zeros(labels.shape, np.int32)
        offset = 0
        for k in range(5):
            components, count = ndimage.label(labels == k, structure=np.ones((3, 3)))
            keep = components > 0
            ids[keep] = components[keep] + offset
            offset += count
    else:
        _, inverse = np.unique(segments, return_inverse=True)
        ids = inverse.reshape(labels.shape) + 1
    counts = np.bincount(ids.ravel())
    candidates = [i for i in range(1, len(counts)) if counts[i] >= minimum]
    candidates.sort(key=lambda i: (-counts[i], i))
    return [ids == i for i in candidates[:maximum]]


def region_pass(solver, problem, labels, levels, masks):
    """Exact full-objective acceptance, with pixel-only and joint coarse proposals."""
    blocks = solver.block_problems(problem)
    labels, levels = labels.copy(), [z.copy() for z in levels]
    current = solver.total_energy(problem, blocks, labels, levels)
    history, tested = [], 0
    for mask in masks:
        best_energy, best = current, None
        cover = [pool(mask[..., None].astype(float), side)[..., 0] >= .5 * side ** 2
                 for side in solver.sides()]
        for label in range(problem.num_labels):
            proposal = labels.copy()
            proposal[mask] = label
            for joint in (False, True):
                coarse = [np.where(m, label, z) if joint else z for m, z in zip(cover, levels)]
                value = solver.total_energy(problem, blocks, proposal, coarse)
                tested += 1
                if value < best_energy - 1e-9:
                    best_energy, best = value, (proposal, coarse, label, joint)
        if best is not None:
            before = current
            labels, levels, label, joint = best
            current = best_energy
            row = dict(before=before, after=current, pixels=int(mask.sum()), label=label,
                       joint_coarse=joint)
            history.append(row)
            emit('region_move', **row)
    return labels, levels, dict(tested=tested, accepted=len(history), moves=history)


def relax_state(solver, problem, labels, levels):
    """Resume measured hierarchy descent from both pixel AND coarse labels."""
    blocks = solver.block_problems(problem)
    labels, levels = labels.copy(), [z.copy() for z in levels]
    current = solver.total_energy(problem, blocks, labels, levels)
    stages, rounds = [], 0
    while True:
        rounds += 1
        before = current
        for level in range(solver.levels + 1):
            base = problem if level == 0 else blocks[level - 1]
            old = labels if level == 0 else levels[level - 1]
            sub = replace(base, unary=base.unary + solver._link_costs(
                level, problem.num_labels, labels, levels))
            with solver_scope(hierarchy_round=rounds, level=level, region_cleanup=True):
                result = (solver.pixel if level == 0 else solver.block_solver).solve(sub, old)
            if level == 0:
                labels = result.labels
            else:
                levels[level - 1] = result.labels
            updated = solver.total_energy(problem, blocks, labels, levels)
            if updated > current + 1e-6:
                raise ValueError('Region cleanup increased hierarchy energy')
            current = updated
            stages.append(dict(round=rounds, level=level, stats=result.stats))
        if current >= before - solver.tol:
            break
    audit = solver.audit(problem, blocks, labels, levels)
    if audit['improving_variables']:
        raise ValueError('Region cleanup failed stationarity audit')
    return labels, levels, current, stages, rounds, audit


def solve_with_regions(solver, problem, initial, segments=None):
    start = time.perf_counter()
    result = solver.solve(problem, initial)
    control_energy = result.energy
    rounds, stages = result.stats['rounds'], result.stats['stages'][:]
    labels, levels, passes = result.labels, result.levels, []
    for _ in range(2):
        masks = region_masks(labels, segments)
        labels, levels, row = region_pass(solver, problem, labels, levels, masks)
        passes.append(row)
        if not row['accepted']:
            break
        labels, levels, value, details, count, audit = relax_state(solver, problem, labels, levels)
        stages.extend(details)
        rounds += count
    blocks = solver.block_problems(problem)
    value = solver.total_energy(problem, blocks, labels, levels)
    audit = solver.audit(problem, blocks, labels, levels)
    if value > control_energy + 1e-6 or audit['improving_variables']:
        raise ValueError('Region method failed objective or stationarity check')
    stats = dict(stages=stages, rounds=rounds, converged=True, audit=audit,
                 stop_reason='budgeted_region_passes_then_single_variable_stationarity',
                 control_energy=control_energy, region_passes=passes,
                 region_scope='At most two passes over the 32 largest regions >=64 pixels; no region-optimality claim')
    return MeasuredResult(labels, value, time.perf_counter() - start, levels=levels,
                          stats=stats, trace=result.trace)


def edit_counts(initial, labels, target):
    valid = target != 255
    changed = valid & (initial != labels)
    wrong, right = initial != target, labels == target
    edge = np.zeros(initial.shape, bool)
    edge[:-1] |= initial[:-1] != initial[1:]
    edge[1:] |= initial[:-1] != initial[1:]
    edge[:, :-1] |= initial[:, :-1] != initial[:, 1:]
    edge[:, 1:] |= initial[:, :-1] != initial[:, 1:]
    distance = ndimage.distance_transform_edt(~edge) if edge.any() else np.full(initial.shape, np.inf)
    result = {}
    for name, mask in [('all', valid), ('boundary_10px', valid & (distance <= 10)),
                       ('interior_over_10px', valid & (distance > 10))]:
        result[name] = dict(changed=int((changed & mask).sum()),
                            fixes=int((wrong & right & mask).sum()),
                            breaks=int((~wrong & ~right & mask).sum()),
                            wrong_to_wrong=int((changed & wrong & ~right & mask).sum()))
    return result

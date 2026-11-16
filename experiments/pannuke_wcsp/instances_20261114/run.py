"""WCSP over fixed HoVer-Net nucleus masks and five foreground types."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from prepare import CACHE, HERE, load, write

N = 5
CLASSES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Non-neoplastic epithelial"]
BETAS = [0.10, 0.25, 0.50, 1.00, 2.00]
UNARIES = ["vote", "mean", "log_mean"]
AREA_CUTS = [24, 48, 80, 128, 192, 320, 512]
AREA_WEIGHTS = [.25, .75]
PRIOR = np.array([[0, 1, 1, 1, 10], [1, 0, 2, 1, 6], [1, 2, 0, 4, 1],
                  [1, 1, 4, 0, 10], [10, 6, 1, 10, 0]], float)


def normalise(table):
    mean = table[~np.eye(N, dtype=bool)].mean()
    return table / mean if mean > 0 else table


def neighbourhood(mask):
    """Symmetric four-nearest graph within 32 pixels, weighted by distance."""
    ys, xs = np.nonzero(mask)
    ids = mask[ys, xs]
    count = np.bincount(ids, minlength=int(mask.max()) + 1)[1:]
    if len(count) < 2:
        return np.array([], int), np.array([], int), np.array([], float)
    centers = np.column_stack([np.bincount(ids, weights=c, minlength=len(count) + 1)[1:] / count
                               for c in (ys, xs)])
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    neighbours = np.argsort(distance, axis=1, kind="stable")[:, :4]
    edges = np.zeros_like(distance)
    row = np.arange(len(count))[:, None]
    edges[row, neighbours] = distance[row, neighbours] <= 32
    a, b = np.nonzero(np.triu((edges + edges.T) > 0, 1))
    weights = np.exp(-distance[a, b] / 24)
    degree = np.bincount(np.r_[a, b], weights=np.tile(weights, 2), minlength=len(count))
    return a, b, weights / np.maximum(1, np.sqrt(degree[a] * degree[b]))


def load_gt(index):
    with np.load(CACHE / f"gt_{index:04d}.npz") as file:
        return file["instances"], file["types"]


def geometry(mask):
    count = np.bincount(mask.ravel(), minlength=int(mask.max()) + 1)
    border = np.unique(np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    valid = ~np.isin(np.arange(1, len(count)), border)
    return np.searchsorted(AREA_CUTS, count[1:], side="right"), valid


def fit(records):
    counts = np.zeros((N, N), int)
    area_counts = np.zeros((len(AREA_CUTS) + 1, N), int)
    for record in records:
        gt, types = load_gt(record["image_index"])
        bins, valid = geometry(gt)
        np.add.at(area_counts, (bins[valid], types[1:][valid] - 1), 1)
        a, b, _ = neighbourhood(gt)
        np.add.at(counts, (types[a + 1] - 1, types[b + 1] - 1), 1)
        np.add.at(counts, (types[b + 1] - 1, types[a + 1] - 1), 1)
    joint = (counts + 1) / (counts + 1).sum()
    marginal = joint.sum(1)
    learned = np.maximum(0, -np.log(joint / np.outer(marginal, marginal)))
    np.fill_diagonal(learned, 0)
    tables = dict(potts=1 - np.eye(N), hybrid=normalise(.8 * normalise(learned) + .2 * normalise(PRIOR)))
    pooled = (area_counts.sum(1) + 1) / (area_counts.sum() + len(area_counts))
    likelihood = (area_counts + 30 * pooled[:, None]) / (area_counts.sum(0) + 30)
    tables["area"] = -np.log(likelihood / pooled[:, None])
    write("tables.json", dict(counts=counts.tolist(), area_counts=area_counts.tolist(),
                              **{k: x.tolist() for k, x in tables.items()}))
    return tables


def unaries(mask, probability):
    k = int(mask.max())
    keep = mask > 0
    ids = mask[keep] - 1
    p = probability[keep].astype(float)
    raw_vote = p.argmax(axis=-1)
    vote = np.bincount(ids * (N + 1) + raw_vote, minlength=k * (N + 1)).reshape(k, N + 1)[:, 1:]
    area = np.bincount(ids, minlength=k)[:, None]
    mean = np.zeros((k, N))
    log_mean = np.zeros((k, N))
    np.add.at(mean, ids, p[:, 1:])
    conditional = p[:, 1:] / np.maximum(p[:, 1:].sum(axis=1, keepdims=True), 1e-12)
    np.add.at(log_mean, ids, -np.log(np.maximum(conditional, 1e-12)))
    # Most frequent non-background pixel type matches HoVer-Net voting
    # for ordinary detections. Resolve all-background detections by pooled TP.
    baseline = vote.argmax(axis=1)
    fallback = vote.sum(axis=1) == 0
    baseline[fallback] = mean[fallback].argmax(axis=1)
    vote = np.where(fallback[:, None], mean, vote)
    cost = {
        "vote": -np.log((vote + .5) / (vote.sum(axis=1, keepdims=True) + N * .5)),
        "mean": -np.log(np.maximum(mean / np.maximum(mean.sum(axis=1, keepdims=True), 1e-12), 1e-12)),
        "log_mean": log_mean / area,
    }
    return baseline, cost, int(fallback.sum())


def solve(unary, edges, pair_cost, beta, baseline):
    """Multistart bidirectional ICM with an independent local-optimum audit."""
    if len(unary) == 0:
        return baseline.copy()
    initial = unary.argmin(axis=1)
    if beta == 0:
        return initial
    a, b, weights = edges
    weights = beta * weights
    adjacency = [[] for _ in unary]
    for i, j, w in zip(a, b, weights):
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    def conditional(i, labels):
        costs = unary[i].copy()
        for j, w in adjacency[i]:
            costs += w * pair_cost[:, labels[j]]
        return costs

    def energy(labels):
        return unary[np.arange(len(labels)), labels].sum() + (weights * pair_cost[labels[a], labels[b]]).sum()

    best, best_cost = None, np.inf
    starts = (baseline, initial) if not np.array_equal(baseline, initial) else (initial,)
    for start in starts:
        labels = start.copy()
        for _ in range(100):
            changed = False
            for order in (range(len(labels)), reversed(range(len(labels)))):
                for i in order:
                    costs = conditional(i, labels)
                    new = costs.argmin()
                    if costs[new] + 1e-12 < costs[labels[i]]:
                        labels[i] = new
                        changed = True
            if not changed:
                break
        value = energy(labels)
        assert value <= energy(start) + 1e-9
        if value < best_cost - 1e-12:
            best, best_cost = labels, value
    costs = unary.copy()
    np.add.at(costs, a, weights[:, None] * pair_cost[:, best[b]].T)
    np.add.at(costs, b, weights[:, None] * pair_cost[best[a], :])
    assert np.all(costs[np.arange(len(best)), best] <= costs.min(axis=1) + 1e-10)
    return best


def confusion(gt, pred):
    return np.bincount((gt.astype(int) * 6 + pred).ravel(), minlength=36).reshape(6, 6)


def iou(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(axis=-1) + cm.sum(axis=-2) - diagonal
    return np.divide(diagonal, union, out=np.zeros_like(diagonal, float), where=union > 0)


def score(cm):
    return iou(cm)[..., 1:].mean(-1)


def patches(records):
    for record in records:
        index = record["image_index"]
        gt, truth = load_gt(index)
        with np.load(CACHE / f"pred_{index:04d}.npz") as file:
            instances, p = file["instances"], file["p"]
        baseline, cost, fallback = unaries(instances, p)
        ground_labels = truth[gt]
        k, g = len(baseline), len(truth) - 1
        cross = np.bincount(((k + 1) * gt.astype(int) + instances).ravel(),
                            minlength=(g + 1) * (k + 1)).reshape(g + 1, k + 1)
        inter = cross[1:, 1:]
        union = cross.sum(1)[1:, None] + cross.sum(0)[None, 1:] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter, float), where=union > 0)
        truth_id, pred_id = np.nonzero(overlap > .5)
        assert len(set(truth_id)) == len(set(pred_id)) == len(truth_id)
        yield dict(index=index, gt=ground_labels, instances=instances, baseline=baseline,
                   graph=neighbourhood(instances), area=geometry(instances), cost=cost, raw=p.argmax(-1),
                   ps=pred_id, ts=truth[truth_id + 1] - 1, match_iou=overlap[truth_id, pred_id],
                   truth=truth[1:] - 1, fallback=fallback)


def evaluate(config, data, tables, persist=False):
    name = config["name"]
    cms, matched, typed_hits, typed_overlap, pred_count = [], [], [], [], []
    changes, fixes, breaks, wrong = 0, 0, 0, 0
    for patch in data:
        base, pred_ids, truth = (patch[key] for key in ("baseline", "ps", "ts"))
        if name == "instance_vote":
            labels = base.copy()
        elif name == "type_oracle":
            labels = base.copy()
            labels[pred_ids] = truth
        else:
            unary = patch["cost"][config["unary"]]
            if config["area_weight"]:
                bins, valid = patch["area"]
                unary = unary + config["area_weight"] * tables["area"][bins] * valid[:, None]
            labels = solve(unary, patch["graph"], tables[config["table"]], config["beta"], base)
        mapped = np.r_[0, labels + 1][patch["instances"]]
        assert np.array_equal(mapped > 0, patch["instances"] > 0)
        cms.append(confusion(patch["gt"], mapped))
        changed = base != labels
        changes += int(changed.sum())
        okay, initial = labels[pred_ids] == truth, base[pred_ids] == truth
        matched.append(int(okay.sum()))
        fixes += int((okay & ~initial).sum())
        breaks += int((~okay & initial).sum())
        wrong += int((~okay & ~initial & changed[pred_ids]).sum())
        typed_hits.append(np.bincount(truth[okay], minlength=N))
        typed_overlap.append(np.bincount(truth[okay], weights=patch["match_iou"][okay], minlength=N))
        pred_count.append(np.bincount(labels, minlength=N))
        if persist:
            path = CACHE / "refined" / name
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / f'{patch["index"]:04d}.npy', labels.astype(np.uint8) + 1)
    cm = np.array(cms)
    truths = np.stack([np.bincount(patch["truth"], minlength=N) for patch in data])
    pq_denom = .5 * (truths + pred_count).sum(0)
    overlap = np.sum(typed_overlap, axis=0)
    pq = np.divide(overlap, pq_denom, out=np.zeros_like(overlap), where=pq_denom > 0)
    matches = sum(len(patch["ps"]) for patch in data)
    result = dict(config=config, bio_miou=float(score(cm.sum(0))), per_class_iou=iou(cm.sum(0))[1:].tolist(),
                  typed_pq=float(pq.mean()), per_class_pq=pq.tolist(), correct_types=sum(matched),
                  matched_type_accuracy=sum(matched) / max(1, matches),
                  changed_nuclei=changes, matched_fixes=fixes, matched_breaks=breaks,
                  matched_wrong_to_wrong=wrong, matched_correct_by_class=np.sum(typed_hits, axis=0).tolist())
    print(name, f'{result["bio_miou"]:.7f}', f'{result["typed_pq"]:.7f}', fixes, breaks, flush=True)
    return result, cm, np.array(typed_overlap), truths + pred_count, np.asarray(matched)


def configs():
    grid = []
    for unary in UNARIES:
        for weight in [0, *AREA_WEIGHTS]:
            betas = BETAS if weight == 0 else [.25, .50]
            choices = [("potts", 0)] + [(table, beta) for table in ("potts", "hybrid") for beta in betas]
            suffix = f"_area{weight:g}" if weight else ""
            for table, beta in choices:
                grid.append(dict(name=f"{unary}_{table}_{beta:g}{suffix}",
                                 unary=unary, table=table, beta=beta, area_weight=weight))
    return grid


def dev(split):
    assert not (HERE / "frozen.json").exists()
    candidates = configs()
    write("protocol.json", dict(
        objective="Sum per-nucleus unary cost plus beta times weighted Potts or learned/biological type-pair costs.",
        beta_grid=BETAS, unaries=UNARIES, area_cost_weights=AREA_WEIGHTS,
        area_bins=AREA_CUTS, area_cost="Smoothed GT area likelihood relative to pooled likelihood. Skip clipped nuclei at image borders.",
        selection="Select maximum development biological mIoU among positive-beta models. Within 0.0001 absolute mIoU of best prefer lower beta, then no area term or smaller area weight, then simpler Potts, then candidate order. Evaluate it and its beta=0 control, regardless of development outcome.",
        oracle="GT type assigned only to uniquely matched detections with IoU >0.5; unmatched detections keep baseline type.",
        primary="Pooled five-type semantic mIoU versus same HoVer-Net instance masks with foreground majority vote.",
        secondary="Pooled typed PQ, matched type accuracy, count of fixes/breaks and reference Raw TP.",
        bootstrap={"replicates": 5000, "seed": 20261115},
        input_ids={group: [r["image_index"] for r in split[group]] for group in ("train", "dev", "test")},
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), HERE / "prepare.py")}))
    tables = fit(split["train"])
    data = list(patches(split["dev"]))
    rows = [evaluate(config, data, tables)[0] for config in
            [dict(name="instance_vote"), dict(name="type_oracle"), *candidates]]
    active = [r for r in rows if r["config"].get("beta", 0) > 0]
    highest = max(r["bio_miou"] for r in active)
    eligible = [r for r in active if r["bio_miou"] >= highest - .0001]
    best = min(eligible, key=lambda r: (r["config"]["beta"], r["config"]["area_weight"], r["config"]["table"] != "potts"))
    chosen = best["config"]
    control = dict(chosen, name=f'{chosen["unary"]}_unary_only', beta=0)
    write("development.json", rows)
    write("frozen.json", dict(wcsp=chosen, unary_control=control, highest_dev_bio_miou=highest))
    print("Frozen", chosen, flush=True)


def final(split):
    assert not (HERE / "evaluation.json").exists()
    frozen = load(HERE / "frozen.json")
    data = list(patches(split["test"]))
    tables_json = load(HERE / "tables.json")
    tables = {name: np.array(tables_json[name]) for name in ("potts", "hybrid", "area")}
    choices = [dict(name="instance_vote"), dict(name="type_oracle"), frozen["unary_control"], frozen["wcsp"]]
    arrays = {}
    rows = {}
    for config in choices:
        result, *inputs = evaluate(config, data, tables, persist=True)
        rows[config["name"]] = result
        arrays[config["name"]] = inputs
    raw = np.stack([confusion(patch["gt"], patch["raw"]) for patch in data])
    arrays["raw_tp"] = [raw]
    matches = [int(len(patch["ps"])) for patch in data]
    gt = sum(len(patch["truth"]) for patch in data)
    pred = sum(len(patch["baseline"]) for patch in data)
    draws = np.random.default_rng(20261115).integers(len(data), size=(5000, len(data)))
    resamples = {name: score(items[0][draws].sum(1)) for name, items in arrays.items()}
    pq_samples = {}
    for name, items in arrays.items():
        if len(items) == 1:
            continue
        overlap, denom = items[1][draws].sum(1), items[2][draws].sum(1) * .5
        pq_samples[name] = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0).mean(1)
    selected, unary = frozen["wcsp"]["name"], frozen["unary_control"]["name"]
    pairs = [(selected, "instance_vote"), (selected, unary), (selected, "raw_tp"),
             (unary, "instance_vote"), ("type_oracle", "instance_vote")]
    comparisons = {}
    matrices = {}
    for name, items in arrays.items():
        matrices[name] = items[0]
    for a, b in pairs:
        difference = resamples[a] - resamples[b]
        left, right = matrices[a], matrices[b]
        patch_deltas = score(left) - score(right)
        comparison = dict(delta=float(score(left.sum(0)) - score(right.sum(0))),
                          ci95=np.percentile(difference, [2.5, 97.5]).tolist(),
                          patch_wlt=[int((patch_deltas > 1e-12).sum()),
                                     int((patch_deltas < -1e-12).sum()),
                                     int((abs(patch_deltas) <= 1e-12).sum())],
                          per_class_iou_delta=(iou(left.sum(0)) - iou(right.sum(0)))[1:].tolist())
        if b in pq_samples:
            comparison["typed_pq_delta"] = rows[a]["typed_pq"] - rows[b]["typed_pq"]
            comparison["typed_pq_ci95"] = np.percentile(pq_samples[a] - pq_samples[b], [2.5, 97.5]).tolist()
        comparisons[f"{a} minus {b}"] = comparison
    np.savez_compressed(HERE / "confusions.npz", **matrices)
    with (HERE / "per_patch.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_index", "ground_truth_nuclei", "predicted_nuclei", "matched_pairs",
                         *matrices, *[f"{name}_matched_types" for name in rows]])
        for i, patch in enumerate(data):
            writer.writerow([patch["index"], len(patch["truth"]), len(patch["baseline"]), matches[i],
                             *[score(matrix[i]) for matrix in matrices.values()],
                             *[arrays[name][3][i] for name in rows]])
    write("evaluation.json", dict(
        rows=rows, comparisons=comparisons, raw_bio_miou=float(score(raw.sum(0))),
        raw_per_class_iou=iou(raw.sum(0))[1:].tolist(), patch_ids=[p["index"] for p in data],
        detection=dict(gt=gt, predicted=pred, matches=sum(matches), missed=gt-sum(matches),
                       unmatched_predictions=pred-sum(matches),
                       detection_f1=2 * sum(matches) / max(1, gt + pred),
                       all_background_vote_fallback=sum(p["fallback"] for p in data),
                       truth_by_class=np.bincount(np.concatenate([p["truth"] for p in data]), minlength=N).tolist(),
                       matched_by_class=np.bincount(np.concatenate([p["ts"] for p in data]), minlength=N).tolist())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "final"])
    split = load(HERE / "split.json")
    {"dev": dev, "final": final}[parser.parse_args().stage](split)

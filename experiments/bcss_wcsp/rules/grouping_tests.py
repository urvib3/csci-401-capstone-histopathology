# Does the U-Net group the right pixels together (class-agnostic), or does it group well
# but name regions wrong? On Ryan's 185 frozen full-res dev inputs.
#
#   boundary F1         predicted vs true class borders, matched within --tol px
#   region coverage     per true region (8-connected piece of one class, >= 64 px): IoU of the
#                       best-overlapping predicted piece, ignoring class; pixel-weighted mean
#   fragmentation       predicted pieces (>= 16 px) per true region; pixel-weighted mean
#   error split         each wrong pixel is a
#                         grouping error  - its true region's majority prediction is correct
#                                           (fixable by making the region coherent), or a
#                         labeling error  - the region's majority prediction is the wrong class
#                                           (grouping can't fix it; needs the right class)
#
#   python experiments/bcss_wcsp/rules/grouping_tests.py --workers 8

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

INPUTS = Path("data/fullres_multiscale_20261115/inputs")
IGNORE, K = 255, 5
EIGHT = np.ones((3, 3))
ARGS = None


def borders(labels, valid):
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= (labels[:-1] != labels[1:]) & valid[:-1] & valid[1:]
    edge[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    return edge


def pieces(labels, valid):
    """8-connected pieces of each class -> (ids, count)."""
    out = np.zeros(labels.shape, np.int64)
    nxt = 0
    for c in range(K):
        comp, n = ndimage.label((labels == c) & valid, structure=EIGHT)
        out[comp > 0] = comp[comp > 0] + nxt
        nxt += n
    return out, nxt


def tile_stats(path):
    z = np.load(path)
    target, pred = z["target"], z["probs"].astype(np.float32).argmax(0)
    valid = target != IGNORE
    s = {}
    tb, pb = borders(target, valid), borders(pred, valid)
    tol = ndimage.generate_binary_structure(2, 1)
    near_t = ndimage.binary_dilation(tb, tol, iterations=ARGS.tol)
    near_p = ndimage.binary_dilation(pb, tol, iterations=ARGS.tol)
    s["bd_tp_p"], s["bd_p"] = int((pb & near_t).sum()), int(pb.sum())
    s["bd_tp_t"], s["bd_t"] = int((tb & near_p).sum()), int(tb.sum())

    true_ids, n_true = pieces(target, valid)
    pred_ids, _ = pieces(pred, valid)
    pred_sizes = np.bincount(pred_ids.ravel())
    cover_w, cover_sum, frag_sum = 0, 0.0, 0.0
    grouping_err = labeling_err = 0
    for r in range(1, n_true + 1):
        region = true_ids == r
        size = int(region.sum())
        cls = int(target[region][0])
        wrong = int((pred[region] != cls).sum())
        majority = np.bincount(pred[region], minlength=K).argmax()
        if majority == cls:
            grouping_err += wrong
        else:
            labeling_err += wrong
        if size < 64:
            continue
        ids, counts = np.unique(pred_ids[region], return_counts=True)
        keep = ids > 0
        ids, counts = ids[keep], counts[keep]
        if not len(ids):
            continue
        best = counts.argmax()
        union = size + pred_sizes[ids[best]] - counts[best]
        cover_sum += size * counts[best] / union
        frag_sum += size * int((counts >= 16).sum())
        cover_w += size
    s.update(cover_sum=cover_sum, frag_sum=frag_sum, cover_w=cover_w,
             grouping_err=grouping_err, labeling_err=labeling_err,
             wrong=int(((pred != target) & valid).sum()), labeled=int(valid.sum()))
    return s


def init(args):
    global ARGS
    ARGS = args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tol", type=int, default=4)
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/runs/fullres_grouping_dev")
    args = parser.parse_args()
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        rows = pool.map(tile_stats, sorted(INPUTS.glob("*.npz")), chunksize=1)
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    precision, recall = t["bd_tp_p"] / max(t["bd_p"], 1), t["bd_tp_t"] / max(t["bd_t"], 1)
    report = {
        "tiles": len(rows), "tol_px": args.tol,
        "pixel_error_rate": round(t["wrong"] / t["labeled"], 4),
        "boundary_precision": round(precision, 4), "boundary_recall": round(recall, 4),
        "boundary_F1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "region_coverage_IoU": round(t["cover_sum"] / t["cover_w"], 4),
        "pieces_per_true_region": round(t["frag_sum"] / t["cover_w"], 2),
        "share_of_errors_grouping": round(t["grouping_err"] / max(t["wrong"], 1), 4),
        "share_of_errors_labeling": round(t["labeling_err"] / max(t["wrong"], 1), 4),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

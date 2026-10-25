# Deliverable 5 — PanNuke WCSP Run and Ground-Truth Evaluation

Ran the WCSP on Aakanksha's full HoVer-Net prediction for PanNuke Fold 2 `image_00748` (164x164, 10,726 active pixels, 47 nuclei) and compared the result against the aligned PanNuke ground truth.

Run from the repository root, in this order:

```bash
python benchmarks/deliverable_5/pannuke_abhishek/run_wcsp.py
python benchmarks/deliverable_5/pannuke_abhishek/evaluate.py
```

`run_wcsp.py` only reads `hovernet_labels_164.npy` and `active_mask_164.npy`. The ground truth is only opened by `evaluate.py`, after the labels are saved.

## WCSP run

Same setup as D4 so the results are comparable:

- Variables: the 10,726 active pixels, each with domain {1, ..., 5}. Inactive pixels stay 0.
- Unary: λ if a pixel changes from its HoVer-Net label, 0 if it keeps it.
- Pairwise: the handout cost table divided by distance, over all five neighbor types (up to 2√2 away). Pairs touching an inactive pixel are left out. That gives 111,755 unique edges.
- Solver: the shared `wcsp/` block scheme with ICM inside each block. The default 100x100 blocks don't fit a 164x164 image, so it uses 60x60 corners and a 44 px cross. No assignments are enumerated.

| Run | λ | Initial objective | Returned objective | Changed pixels | Rounds | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unary-only | 1 | 0.00 | 0.00 | 0 | 1 | 1.8 s |
| Strong fidelity | 139 | 772.81 | 772.81 | 0 | 1 | 1.8 s |
| Weaker fidelity | 1 | 772.81 | 723.15 | 13 | 2 | 3.4 s |

Why λ = 139 is strong enough: a pixel has at most 24 neighbors, and the sum of 1/d over them is 13.82. The biggest table cost is 10, so changing one pixel can save at most 138.2 in pairwise cost. Changing any k pixels saves at most 138.2k but costs λk, so with λ > 138.2 no change can lower the objective. This bound is actually reached on this mask, since some pixels have all 24 neighbors active. The run also confirms it empirically: 0 changes.

ICM is greedy, so the results are local optima, not proven global ones. For every run, no single pixel can lower the objective by changing its label, and solving the whole image as one block gives the same weak result (723.1453).

## Evaluation

The ground truth for this patch only has one class: all 9,250 nucleus pixels are Inflammatory. I checked this against the raw `masks.npy` channels, so it is not an alignment bug.

Strong fidelity output is identical to HoVer-Net on every pixel.

The 13 pixels changed in the weaker fidelity run:

| Change | Pixels | What changed |
| --- | ---: | --- |
| Wrong → correct | 0 | |
| Correct → wrong | 6 | Inflammatory → Connective |
| Wrong → different wrong | 0 | |
| Ground truth is background | 7 | 5 Connective → Inflammatory, 2 Inflammatory → Connective |

Before and after:

| Metric | HoVer-Net | WCSP (λ = 1) | Difference |
| --- | ---: | ---: | ---: |
| Class agreement on overlap (8,229 px) | 0.7737 | 0.7730 | −0.0007 |
| Accuracy on HoVer-Net active region | 0.5936 | 0.5930 | −0.0006 |
| Accuracy on ground-truth nucleus region | 0.6883 | 0.6877 | −0.0006 |
| Pixel accuracy, full patch | 0.8000 | 0.7997 | −0.0002 |
| Inflammatory IoU | 0.6399 | 0.6391 | −0.0008 |

Error breakdown:

| Error | Pixels | Can the WCSP fix it? |
| --- | ---: | --- |
| Detection: ground-truth nucleus missed by HoVer-Net | 1,021 | No, never a variable |
| Detection: HoVer-Net nucleus where ground truth is background | 2,497 | No, the WCSP can't relabel to background |
| Classification: wrong class on overlap, HoVer-Net | 1,862 | Yes |
| Classification: wrong class on overlap, after WCSP | 1,868 | |

What we see:

- The pipeline works end to end on a real full prediction: HoVer-Net → WCSP → ground-truth comparison.
- Strong fidelity reproduces HoVer-Net exactly, as expected.
- At λ = 1 the WCSP lowers the objective but makes accuracy slightly worse. It made 0 corrections and 6 new errors. A lower objective only means the labels fit the cost table better, not that they are more correct.
- Almost all of HoVer-Net's class errors are whole nuclei labeled Connective (3,231 px) when they are Inflammatory. The WCSP can't fix a whole wrong nucleus because every pixel inside it agrees with its neighbors, so it only moves pixels on the edges where two nuclei touch.
- Same thing we saw on BCSS: the pairwise terms only reach 2 pixels, so they only change boundaries.
- This is one patch with one ground-truth class, so it doesn't say much about PanNuke in general. The next step is running more patches. Making the decision per nucleus (using the instance map) instead of per pixel would also probably fit PanNuke better.

How it was evaluated:

- Overlap = pixels that are nucleus in both HoVer-Net and the ground truth. This is the same 0.7737 class agreement Aakanksha reported, so the evaluation matches her preparation.
- Active region accuracy counts ground-truth background as wrong. Ground-truth nucleus accuracy counts missed pixels as wrong.
- IoU is only reported for Inflammatory since it is the only class in the ground truth. The other predicted classes have IoU 0 both before and after, and Dead is never predicted.
- λ = 1 was taken from D4 and not tuned on this patch, since tuning it here would use the ground truth.

Files:

- `run_wcsp.py`, `results.json`: WCSP runs, settings, objectives, energy trace per round
- `wcsp_strong_labels.npy`, `wcsp_weak_labels.npy`: returned 164x164 label maps (0 = inactive)
- `evaluate.py`, `evaluation.json`: all metrics above plus confusion matrices

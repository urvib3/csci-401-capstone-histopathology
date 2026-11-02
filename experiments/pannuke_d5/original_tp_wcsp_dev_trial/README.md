# WCSP with original pixel type probabilities

All 26,896 pixels in each central 164 × 164 crop are variables with six possible labels, including background. The original six-channel HoVer-Net TP softmax supplies unary costs: `-log(max(P(class), 1e-8))`, minus each pixel's minimum (an energy constant). No NP multiplication, nucleus voting, clustering or class-transition prior is used. Initialization is the raw TP pixel argmax. Ground truth is loaded only after optimization for evaluation.

Neighbor costs remain fixed from the preceding experiments: the existing biological compatibility table, background/type cost 0.25, and 24 neighbors across five distance groups, weighted by inverse distance. These are experimental settings, not learned pixel interaction costs. See `audit.json` for the full table.

| Method | Mean biological mIoU | Mean nucleus/background IoU |
|---|---:|---:|
| Saved HoVer-Net | 0.474933 | 0.746156 |
| Original TP pixel argmax | 0.438640 | 0.732753 |
| Original TP probabilities + WCSP | 0.440987 | 0.723412 |

These are the same 15 development patches; mIoU averages biological classes present in each ground-truth patch, then averages across patches. WCSP improves mIoU on 8 patches and worsens 7 versus raw TP argmax. It fixes 1,162 pixels but introduces 1,101 errors relative to that raw readout. Relative to saved HoVer-Net, 2 patches improve and 13 worsen. This formulation does not improve the saved baseline, and binary nucleus segmentation gets worse even against raw TP argmax.

All 15 optimizations reduce or retain energy and reach a single-pixel local minimum. This does not guarantee global optimality or improved ground-truth accuracy. The result tests this fixed graph/cost configuration, not every possible probability-based WCSP formulation. The saved baseline also uses instance postprocessing, unlike this pixel-only experiment.

```sh
PYTHONPATH=/private/tmp/pannuke-eval-runtime python3 experiments/pannuke_d5/run_unified_dev.py \
  --probability-dir experiments/pannuke_d5/hovernet_raw_head_dev_maps \
  --probability-head original_tp --initialization probability_argmax --skip-type-prior \
  --output /tmp/pannuke_original_tp_wcsp
```

`summary.json` contains aggregate comparisons against saved HoVer-Net; `per_patch.csv` contains patch metrics; `audit.json` and each patch's `wcsp_vs_raw_evaluation.json` compare against raw TP argmax.

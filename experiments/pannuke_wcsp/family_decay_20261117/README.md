# PanNuke constraint-family decay

The earlier block-only test fixed the pixel and coarsest sibling weights
at 0.30, then weakened finer block sibling costs. This follow-up varies
three families separately and then combines their development choices.
The pixel fidelity term, graph, label tables and ICM optimizer are reused.

| Family | Variation | Anchor or control |
|---|---|---|
| Sibling | Block weight proportional to block side raised to a positive or negative power | Same mean across block levels; pixel sibling weight stays at 0.30 |
| Parent–child | Disagreement weight proportional to child side raised to a power, including area scaling | Same mean across links between levels; weaker and stronger uniform controls |
| Block fidelity | Confidence cost weighted by level, with an optional multiplier based on the top-two mean probability margin in that block | V1 has no block unary, V2 uses uniform 0.30, and lower uniform 0.15 is also tested |

## Frozen weights for the second decay experiment

Each individual test changes one column from its original settings. The combined run uses all three columns below together. The values are rounded to six decimals. [Exact settings](frozen.json)

| Level | Sibling coefficient | Parent–child penalty (this level as child) | Fidelity weight |
|---|---:|---:|---:|
| 1×1 pixel | 0.300000 | 0.016129 | 1.00 |
| 2×2 block | 0.774194 | 0.032258 | 0.30 × confidence factor |
| 4×4 block | 0.387097 | 0.064516 | 0.30 × confidence factor |
| 8×8 block | 0.193548 | 0.129032 | 0.30 × confidence factor |
| 16×16 block | 0.096774 | 0.258065 | 0.30 × confidence factor |
| 32×32 block | 0.048387 | — | 0.30 × confidence factor |

Confidence factor = **0.25 + 0.75 × margin**, where the margin is the difference between the two highest mean HoVer-Net probabilities in that block. At margin 0 the fidelity weight is 0.075. At margin 0.60 it is 0.210.

- **Sibling baseline:** 0.30 at every level. An edge uses the listed coefficient divided by 4, times its label-pair cost.
- **Parent–child baseline:** 0.10 per mismatch at every link. The row gives the cost from that child up to the next level, so 32×32 has no parent.
- **Fidelity baselines:** pixels keep weight 1.00. V1 has zero block fidelity and V2 has 0.30 at every block level. The adaptive run uses V2 as its matched control.

Holding the arithmetic mean across levels fixed controls the average
coefficient, not the sum of edge weights. Finer levels contain more edges.
The area schedule adapts BCSS's use of child area to the PanNuke weight
range. It is not a direct copy of BCSS's full objective.

- **Development:** 12 original hierarchy development patches and a seeded
  sample of 36 previously inspected nucleus-experiment patches. The original
  V1/V2 implementations are checked for label equality on two dev patches.
- **Comparison:** 300 patches disjoint from this sweep's development IDs.
  These are the cached type-set replication patches and their previous
  type-set outcomes are already known. Treat the result as exploratory
  transfer on previously inspected patches.
- **Selection:** choose the best dev setting in each family, freeze them
  and the combined setting, then evaluate all alongside Raw TP, V1, V2,
  the previous sibling decay and matched pixel/block Potts.
- **Evaluation:** pooled biological mIoU, pairwise patch bootstrap intervals,
  patch wins/losses/ties, pixel fixes/breaks, and a Dead-excluded sensitivity.
  All hierarchy arms use a single-variable local optimum audit.

Run from this directory with the existing inference caches:

```sh
../../../.venv/bin/python run.py
```

See the [protocol](protocol.json), [frozen settings and development scores](frozen.json),
and [paired results](evaluation.json).

On the 300-patch comparison, all graded and adaptive hierarchy gains versus
Raw TP had paired intervals including zero. The joint schedule changed mIoU
by -0.00001953 versus V1, with interval [-0.00008729, +0.00004660].

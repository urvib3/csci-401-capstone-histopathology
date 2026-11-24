# Multi-scale block WCSP: progress

Updated by Claude as work goes on. Started 2026-11-08.

Setup (same as Deliverable 5): pretrained U-Net, original tiling, central 256 × 256,
**data-derived b** (`configs/tables/bcss_elicited.csv`), λ = 8, ICM with the 100/56/100 block scheme.
Settings are chosen on dev (185 tiles: U-Net mIoU 0.7067, current data-derived WCSP 0.7090, +0.0023)
and tested once on val (2,124 tiles: U-Net 0.6489, current data-derived WCSP 0.6506).
The Deliverable 6 181-tile set is in `/scratch2/rnene`, which we can't read, so
Deliverable 5's set is the one we can reproduce exactly.

- [x] Implement block levels (`wcsp/solve/hierarchy.py`) + solver config (`configs/bcss_multiscale.yaml`)
- [x] Unit tests (`tests/wcsp/test_hierarchy.py`, 5 pass): energy matches brute force, never increases,
      μ = 0 gives the pixel solution, huge μ makes blocks uniform
- [x] Smoke test, 8 dev tiles: runs; μ = 0 matches Deliverable 5 labels (one tile slightly lower energy
      from extra outer rounds, so `msd_mu0` is the like-for-like control)
- [x] Diagram (`experiments/bcss_wcsp/figures/multiscale_diagram.png`), re-render with the data-derived table at the chosen μ
- [x] ~~Handwritten-b sweep~~: switched to data-derived b at the user's request and cancelled it
      (5/10 had finished: μ = 0 → 0.2 gave dev mIoU 0.7098 → 0.7104; kept only as a side note)
- [x] **Dev sweep, data-derived b** (185 tiles; U-Net 0.7067, current WCSP 0.7090): μ = 0 0.7090 (control matches),
      0.05 0.7093, 0.1 0.7097, 0.2 0.7101, **0.5 0.7108**, 1 0.7106. At μ = 0.2: 1 level 0.7097, 2 levels 0.7100,
      3 levels 0.7101; no block unary 0.7105; no block neighbors 0.7096.
      **Frozen: μ = 0.5, 3 levels.** Paired vs. current WCSP on dev: +0.0018 [+0.0008, +0.0027]
- [x] Error-structure analysis on val (`experiments/bcss_wcsp/analysis/error_structure.py`):
      82% of U-Net errors sit in predicted regions ≥ 8,192 px; 37% are ≥ 20 px from any boundary;
      an oracle that perfectly fixes every island < 64 px only reaches the current WCSP's mIoU
      (rerun on the data-derived val run too)
- [x] **Deeper levels**: level 4 (32 px blocks) at μ = 0.1 / 0.2 / 0.5: 0.7097 / 0.7101 / 0.7108, the same as 3 levels;
      level 5 (64 px) output identical to level 4. Larger blocks add nothing.
- [x] Comparison figure (`figures/multiscale_compare.png`): image | U-Net | old WCSP | new WCSP | ground truth | diff,
      2 val tiles where it helps most, 1 where it hurts most
- [x] Region statistics (`analysis/region_stats.py`): after the WCSP, region counts and island sizes already
      match the ground truth; what stays wrong is *which* classes border / co-occur
      (e.g. P(inflammatory in tile | necrosis) 0.01 in truth vs 0.18 after WCSP)
- [x] Class-presence ceiling (`analysis/class_presence.py`): 23% of the WCSP's val errors are pixels given a class
      the tile's ground truth doesn't contain; fixing only those would give val mIoU 0.7532 (vs. 0.6506)
- [x] Val (2,124 tiles): mIoU 0.6517 vs. current WCSP 0.6506; paired +0.0011 [+0.0004, +0.0017]; 32 vs 10 s/tile
- [x] **Convergence check:** old WCSP run to convergence (≤ 16 rounds) stays at val 0.6506 / dev 0.7090;
      multi-scale with 20 outer rounds stays at dev 0.7108. Multi-scale vs. converged old WCSP on val: +0.0011
      [+0.0004, +0.0017]. The gain is not from extra solver time.
- [x] Deliverable 6: "Multi-scale blocks" + "What the WCSP can and can't fix" sections added (not pushed)
- [x] General evaluation of the WCSP approach + better options (in the Deliverable 6 section)

## Biological rules (started 2026-11-09)

- [x] Rules checked against fit-split ground truth (see Deliverable 6). Kept: tile-level co-occurrence,
      necrosis never borders inflammatory / other. Dropped: "necrosis needs tumor" (only 58% of necrosis tiles).
- [x] Presence solver (`wcsp/solve/presence.py`): choose the tile's class set exactly from the U-Net evidence
      + γ · −log P(set) from fit counts, then the usual WCSP with other classes forbidden. 6 tests pass.
- [x] Exclusion tables: necrosis–inflammatory and necrosis–other b raised to 15 and 50.
- [ ] **Running:** fit-split prior (job 13892100) → presence dev sweep γ ∈ {0, 10, 30, 100, 300, 1000, 3000}
      (13892101); exclusion dev runs (13892102)
- [ ] Combine the best rule with multi-scale; freeze; one val run
- [ ] Charts + concise rewrite of my Deliverable 6 sections; push

## Full-512 follow-up audit — 2026-11-22

The earlier checklist describes historical pretrained/central-256 work. Its
"running" labels are historical notes, not a current scheduler check. Those
scores and oracle ceilings must not be treated as measurements of the current
full-512 checkpoint or its cohort.

### Submitted on Discovery

- Sweep array **14477362**, account `dilkina_1463`, partition `main`:
  15 uncapped lambda={4,8,16} x mu={0,0.5,1,2,4} treatments and two
  lambda=8, mu={0,1} outer-curve treatments with 4/8/12-round checkpoints.
- 185 frozen development tiles per treatment; 3,145 tile solves. Four array
  tasks run concurrently, each with eight single-threaded workers.
- Dependent report **14477363** uses `afterany` so partial results are reported
  even when a task fails or reaches its allocation limit.
- Protocol/output root: `data/fullres_expanded_20261122_v2/`. Launch verification
  found 32 worker logs, live traces, completed audited tiles and no tracebacks.
- Only this sweep and report were submitted. The candidates below are proposals.

**Subsequent authorized launch:** the 40-treatment correction follow-up is now
implemented and submitted on Discovery: preparation/full-512 smoke job
**14477975**, dependent correction array **14477976**, report **14477977**.
The array covers 7,400 development tile solves and starts only after the
preparation and smoke checks pass. The exact configurations, fit-only
appearance classifier, region-move budget, controls and reproduction commands
are documented in the [BCSS README](../README.md#full-512-correction-follow-ups).
Outputs and submission records live in `data/fullres_corrections_20261122/`.
The recommendations below retain their rationale; secondary candidates not
listed in that 40-treatment table remain proposals.

### Does the U-Net need a higher temperature?

The implementation already supports `p_T = softmax(log(p)/T)` in
[`unary.py`](../../../wcsp/model/unary.py). Away from probability clipping,
the cost gap between two pixel classes is divided by T. For example, changing
a 90/10 prediction costs log(9)=2.197 at T=1 and 1.099 at T=2. The argmax
baseline does not change. This makes a higher temperature a useful test of
weaker fidelity, even when it does not improve probability calibration.

A dev-only check sampled up to 2,000 labeled pixels per frozen tile with seed
0 (367,552 pixels total) and used the existing `fit_temperature` implementation.
Among its 41 log-spaced temperatures from 0.25 to 8 plus explicit
{1,1.5,2,3,4,6,8}, **T=1 minimized sampled negative log-likelihood**:

| T | Sampled NLL |
|---|---:|
| 1 | 0.501001 |
| 1.5 | 0.546415 |
| 2 | 0.633104 |
| 3 | 0.805417 |
| 4 | 0.939359 |

Of 67,306 sampled argmax errors, 9,803 (14.6%) had >=90% confidence. This
does not establish global overconfidence or identify which errors a WCSP
can repair. Calibration NLL and correction mIoU answer different questions.
No probabilities, protocol settings, or running experiments were changed.

For pixel-only energies, temperature and scalar unary weight overlap:
`E = (alpha/T) U + lambda V`, up to clipping. In the hierarchy, coarse
probabilities are reconstructed from pixel unaries before pooling, so alpha
or T also changes block evidence; neither is a pure pixel-fidelity ablation.
Use `block_alpha` to independently vary the strength of the pooled block
unary. The generic `wcsp` pipeline takes temperature from
`classifier.temperature` and overwrites `problem.temperature`; the frozen
full-resolution runners instead build directly from their problem settings.

### Inventory of written plans and transferable methods

| Candidate | Existing evidence / implementation | Remaining work for current BCSS corrections |
|---|---|---|
| Temperature or weaker unary fidelity | Calibration/unary code exists; no temperature treatment in the running grid | Dev-only T={1,1.5,2,3,4} pilot with structural weights fixed; distinguish NLL calibration from correction selection |
| Weaker block fidelity, separate weights at each scale | Historical BCSS no-block-unary ablation; PanNuke family-decay experiments | Matched `block_alpha` ablation; then separate scale weights if justified |
| Confidence-dependent fidelity or class-specific penalties | PanNuke confidence-weighted block unary and edit gates; class-specific parent-child costs proposed | Adapt to BCSS; verify rare-class effects and confident errors, not just low-margin edits |
| Tile class-presence/co-occurrence prior | BCSS presence solver, fit-prior script, seven gamma configs and prior artifact exist; TODO explicitly proposes combining with multiscale | Refit/check priors against current fit split and tile field; retune gamma; compare presence alone, hierarchy alone, combination |
| Biological exclusion costs | BCSS exclusion tables with entries 15/50 and dev configs exist | Test on current frozen dev inputs with matched controls; inspect fake boundaries and rare-class deletion |
| Region-label WCSP over superpixels/RAG | [Design](../../../docs/design/segmentation-wcsp-pipeline.md) proposes region labels, boundary keep/merge multicut, boundary-band refinement and region-then-boundary hybrid; marked unimplemented | Implement image-derived regions and aggregated costs; retain small structures; compare region-only and hybrid |
| Region-wide or fusion proposal moves | New search proposal; existing descent audits only single-variable stationarity | Propose coordinated connected-region flips or fusions, accept energy-lowering moves, compare equal runtime and semantic gains |
| Robust higher-order region consistency | Research extension of the region plan; no current implementation | Soft/truncated region agreement so uncertain segments can split without forcing every pixel to one type |
| Class-area/proportion/cardinality priors | PanNuke type-count and type-set code; count priors did not beat voting there | Adapt to tissue area fractions; preserve plausible rare classes; do not copy nucleus-count priors directly |
| Image-aware boundaries and informative nonlocal links | RGB/HED contrast and distant edges already implemented; old random distant pairs gave no gain | Match links using appearance/embeddings and normalize degree; use independent image evidence rather than random extra smoothing |
| Richer regional unaries | [Earlier handoff](../../../benchmarks/bcss/HANDOFF_unsupervised_pixel_classification.md) proposes HED+texture, GMM probability unaries, small cluster-naming classifier and Phikon-v2 comparison | Add complementary appearance evidence to U-Net region costs; compare against each unary source alone |
| Scale/context-specific learned compatibility | BCSS data-derived pixel table exists; PanNuke has scale-specific tables | Fit on region/scale-appropriate training relations, with shrinkage and matched Potts controls |
| Wider slide context | Region design explicitly considers stitched-ROI inference and cut segments | Build tile-neighbor context from coordinates; test boundary artifacts and larger-field presence priors |
| Structured learning of correction costs | Old CE+expected-pairwise training arms gave no gain over CE | A genuinely different follow-up could learn WCSP weights/unroll correction; larger implementation effort |

The PanNuke [type-set replication](../../pannuke_wcsp/presence_scale_20261114/README.md)
is the strongest existing context lead: on 300 patches it improved biological
mIoU by 0.00249 over fixed-mask voting, with a paired interval excluding zero.
It is evidence for that nucleus formulation, not a measured BCSS improvement.
The [family-decay study](../../pannuke_wcsp/family_decay_20261117/README.md)
found no clear benefit from its graded/adaptive hierarchy schedules. The
[scaling report](../../pannuke_wcsp/scaling_siblings_20261109/REPORT.md)
still proposes mixed-class block summaries and class-specific parent-child
costs to protect small/rare structures.

### Recommended next sequence

1. Finish the submitted grid. Run a small, separately frozen temperature and
   block-fidelity pilot on dev, retaining T=1 and unchanged-block controls.
2. Revisit BCSS class-presence plus hierarchy, using current fit-split priors.
   The existing presence method selects an allowed set from unary evidence
   plus its prior before spatial solving; this does not jointly optimize the
   full spatial energy. Joint set/region optimization is a separate extension.
3. Add coordinated region proposals to distinguish an ICM energy barrier from
   a misspecified objective. A region can be single-pixel-stable even when a
   coordinated relabeling would lower energy. More iterations alone cannot
   escape that state. Lower energy still does not guarantee better mIoU.
4. Introduce independent image/feature-based regional evidence and robust
   region constraints if the objective continues to prefer wrong regions.

For every candidate report pooled/per-class IoU, wrong-to-right fixes,
right-to-wrong breaks, wrong-to-other-wrong edits, interior-region versus
boundary changes, ROI-paired intervals, runtime and energy. A larger edited
fraction alone is not success. Select on dev and freeze before new held-out
evaluation. Store any new protocol in a separate output root; keep the
running sweep's source hashes and frozen inputs intact.

Do not prioritize deeper grids or random distant edges without new evidence:
historical BCSS deeper levels added no gain and random distant edges were
slower without a gain. Unlimited Toulbar2 also failed to complete the older
benchmark within its allocation; it is not a demonstrated accuracy solution.

### Research references for extensions

- [Guo et al., temperature calibration](https://proceedings.mlr.press/v70/guo17a.html).
- [Lempitsky et al., fusion moves](https://www.robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10a/Lempitsky10a.pdf).
- [Kohli et al., robust higher-order region consistency](https://ora.ox.ac.uk/objects/uuid%3Af5b8b0ce-cb7d-4c41-b10a-dbdc03d813df).
- [Delong et al., label costs](https://www.csd.uwo.ca/~yboykov/Abstracts/cvpr10-abs.html).
- [Boykov et al., graph-cut move making](https://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf).
  Standard alpha-expansion guarantees require metric compatibility costs.
  A direct triangle-inequality check found 2 violating ordered triples in
  the current data-derived table and 12 in the handwritten table. Do not
  assume standard metric guarantees for either; consider compatible fusion
  solvers, exact small-region moves, or an explicitly evaluated metric model.

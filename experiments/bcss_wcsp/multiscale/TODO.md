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
- [ ] **Convergence check (running):** current WCSP with 50 rounds on dev + val (jobs 13860296, 13860298);
      multi-scale with 20 outer rounds on dev (13860297). Does the old WCSP catch up with more solver time?
- [x] Deliverable 6: "Multi-scale blocks" + "What the WCSP can and can't fix" sections added (not pushed)
- [x] General evaluation of the WCSP approach + better options (in the Deliverable 6 section)

# Adrenal gland: decision log

Every choice made for the Adrenal per-organ run, in order, with the reason. All on 2026-11-22.

## Scope and protocol

| # | Decision | Why |
|---|---|---|
| 1 | Split organs across the team: Breast (Kashvi), Colon (Aakanksha), Adrenal gland (Abhishek). | The three largest PanNuke organs by patch count in the earlier 300-patch sample. |
| 2 | Run **all 8 WCSP families** per organ, not only the type-set WCSP: pixel learned table, hierarchy V1/V2, scaling + siblings, decay (sibling decay and per-constraint schedules), nucleus pairwise, type counts, type set, Raw TP component. | Instruction: "run everything; see if one experiment works." Decay and nucleus-level runs were flat on mixed organs in Deliverable 6, but organ-specific tuning is the one place they could still differ. |
| 3 | One shared script, `organ_wcsp/run_all.py`, imports the original objectives and solvers unchanged. | Avoids three slightly different reimplementations. The original folders hardcode their splits and refuse reruns, so they are imported, never edited. |
| 4 | Pin the environment to the versions the original runs recorded: Python 3.12, numpy 2.2.6, torch 2.12.0, tiatoolbox 2.1.2, numba 0.65.1. | The HoVer-Net loader asserts tiatoolbox source-file hashes. Verified that the fold 2 archive, checkpoint and architecture hashes match the recorded provenance. |
| 5 | Protocol: exclude the 1,235 fold 2 IDs used before; shuffle with `default_rng(20261122)`; 40% fit / 20% dev / 40% test; fit tables on fit ground truth only; tune on dev; freeze; score test once. | The same protocol was given to Kashvi and Aakanksha, so per-organ results line up. |
| 6 | "Works" = Holm-adjusted (8 families) two-sided paired-bootstrap p < 0.05 against the family's matched control, with a positive gain. Also report organ-tuned vs. the original global frozen config. | Running 8 families raises the chance that one looks significant by luck. |
| 7 | Matched controls: Raw TP for pixel table, hierarchy and Raw TP component; organ V2 for scaling + siblings; organ V1 for decay; fixed-mask vote for nucleus pairwise, type counts and type set. | Each family is compared against the method its change is layered on, as in the original experiments. |
| 8 | Pixel learned table runs on full 256×256 patches, not the central 164×164 crop from Deliverable 5. | Every other family uses full patches from the same HoVer-Net inference. One input keeps all comparisons paired. |
| 9 | Global baselines: each family's original frozen config with its original all-organ tables. For decay, the "combined" schedule from `family_decay_20261117`. For hierarchy, V1 from `hierarchy_20261109`. | These are the latest frozen settings in each experiment folder. |

## Data

| # | Decision | Why |
|---|---|---|
| 10 | **Adrenal is too small on fold 2 alone**: 148 patches, 75 already used, which leaves only 29 for test. Abhishek chose option 2: use **folds 1, 2 and 3** for everything. | 29 test patches would give intervals too wide to detect gains of the size seen before (+0.002 to +0.005). No experiment in the repo had used folds 1 or 3. |
| 11 | Patch IDs: fold 2 keeps its index, fold 1 = 10000 + index, fold 3 = 30000 + index. Exclusions apply to fold 2 only. | Keeps every earlier exclusion list valid and avoids cache-name collisions. |
| 12 | Each fold is shuffled and split 40/20/40 separately, then pooled. Fold 2 is processed first with the same seed. | Same fold mix in fit, dev and test. A fold-2-only run (Breast, Colon) reproduces the plain protocol split exactly. Verified: identical fit/dev/test IDs for Adrenal fold 2. |
| 13 | The first fold 3 download was truncated (616 of 718 MB). Resumed it, then verified both new archives with `unzip -t`. SHA256: fold 1 `6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b`, fold 3 `503456d669bc03f04c1825504d9e5f0bed6c4fe9029846fafbbb313bf8a8b7da`. | No earlier run recorded hashes for folds 1 and 3. These are the first. |
| 14 | Final split: fold 1 54/27/53, fold 2 29/15/29, fold 3 62/31/62, for **145 fit / 73 dev / 144 test**. The fold-2-only split is kept in `superseded_fold2_only/`. | Five times the fold-2-only test set. |
| 15 | Report results **per fold** as well as pooled. | It's unknown which folds the pretrained HoVer-Net was trained on. If it saw folds 1 or 3, raw scores there would be inflated. WCSP-vs-control gains are paired on identical predictions, so they stay comparable. |
| 16 | The type-set prior merged sparse nucleus-count bands (under 10 fit patches each), leaving cuts at 8 and 20 instead of 8, 20 and 40. | The protocol rule. Few Adrenal patches have 40 or more nuclei. |

## Running

| # | Decision | Why |
|---|---|---|
| 17 | Smoke test on 2 dev patches (no tuning, nothing frozen) before the real run. | Confirmed all 8 method types run with both global and organ tables. Test patches were not touched. |
| 18 | The first dev run crashed loading Numba's on-disk cache, which was written by the smoke test. Registered the imported modules in `sys.modules` and deleted only the 4 cache files the smoke test had created (gitignored). Then reran dev. | The crash happened before any scoring finished. No `frozen.json` existed, so nothing was selected twice. |
| 19 | Dev selected type set with strength 4 and pixel table with β = 0.5, both at the edge of their grids. **Kept them as frozen** rather than widening the grids. | Widening a grid after seeing dev results is fine in principle, but the protocol shared with Kashvi and Aakanksha fixes the grids. Changing them for one organ would break comparability. Flagged as a limitation. |
| 20 | Test was scored exactly once. | Protocol. `run_all.py` refuses to rerun once `evaluation.json` exists. |

## Reporting

| # | Decision | Why |
|---|---|---|
| 21 | Headline: **no organ-tuned family works on Adrenal**. | Every Holm-adjusted p = 1.00. |
| 22 | Also report that the **global type-set config improves Adrenal** (+0.0049 [+0.0016, +0.0089], Holm p = 0.0096 across the 8 global comparisons), and say plainly that this Holm step was added after seeing results. | The global configs were fixed before test and their comparisons were already part of `evaluation.json`. Only the multiple-comparison adjustment for them came afterwards. |
| 23 | Point out that on Adrenal fixed-mask vote is *below* Raw TP, so the global type set is only about +0.0005 over Raw TP. | Without this, "beats vote" reads as "beats HoVer-Net", which it barely does here. |

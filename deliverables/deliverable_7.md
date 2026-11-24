# Deliverable 7

**Web version:** https://urvib3.github.io/csci-401-capstone-histopathology/deliverable7/

*November 23, 2026 · PanNuke (cell nuclei) and BCSS (tissue regions)*

---

## Recap of results

mIoU, with the change vs. the base model in brackets. "—" = not run (yet).

**BCSS protocol in this recap:** the earlier pretrained U-Net / central-256
evaluation. The trained **full-512** model and its correction experiments are
reported [separately below](#bcss-full-resolution-correction-experiments);
their absolute scores use a different model and prediction field.

| Method | PanNuke · Breast test | BCSS · dev | BCSS · val |
|---|:---:|:---:|:---:|
| Base | 0.4323 | 0.7067 | 0.6489 |
| Pair (HW) | — | 0.7097 (+0.0030) | 0.6507 (+0.0019) |
| Pair (DD) | 0.4324 (+0.0001) | 0.7090 (+0.0023) | 0.6506 (+0.0018) |
| Hier | 0.4323 (0.0000) | 0.7107 (+0.0040) | 0.6517 (+0.0028) |
| Hier+Decay | 0.4323 (0.0000) | 0.7108 (+0.0041) | 0.6517 (+0.0029) |
| Cluster | 0.4344 (+0.0020) | 0.7163 (+0.0095) | 0.6544 (+0.0055) |
| **Presence** | — | **0.7229 (+0.0162)** | **0.6550 (+0.0061)** |
| **Pres+Clust** | **0.4484 (+0.0161)** | **0.7242 (+0.0175)** | **0.6564 (+0.0075)** |

**Base** HoVer-Net / U-Net · **Pair (HW / DD)** pixel WCSP, handwritten / data-derived table · **Hier** multi-scale blocks · **Hier+Decay** with level decay · **Cluster** one label per nucleus (PanNuke) or SLIC superpixel (BCSS) · **Presence** drop classes that shouldn't be in the tile · **Pres+Clust** both

![Results by method](figures_20261123/gains_both_datasets.png)

**Takeaway:** smoothing (pixel, hierarchical, decay) barely moves either dataset. Presence, which removes classes that shouldn't be in a tile, gives the only clear gain on both.

<sub>PanNuke: `experiments/pannuke_wcsp/organ_wcsp/breast/evaluation.json` (branch `organ-wcsp-breast`); there, "cluster" means one label per detected nucleus. BCSS: run summaries, λ = 8, SLIC superpixels for "cluster".</sub>

---

## Problems

**The U-Net has limitations, and WCSP right now is only a smoother, so it can't fix them.** Smoothing cleans up noisy borders; the real errors are bigger.

| Problem | Dataset | What goes wrong | What could fix it |
|---|---|---|---|
| **Segmentation** | BCSS | One true region is broken into many pieces | Train the U-Net to keep regions whole |
| **Class identification** | BCSS | A whole region gets the wrong class, or a class that isn't in the tile | Strong data-derived rules (e.g. presence), not light smoothing |
| **Mixed tissue types** | PanNuke | Patches from all organs share one model and one prior, so no organ is fit well | Run per tissue type: fit and tune on one organ at a time |

![BCSS examples of the two problems](figures_20261123/issue_examples.png)


---

## PanNuke

### Preliminary experiments

*Frozen HoVer-Net · 400 dev patches (folds 1 and 3, all organs except Adrenal) · no training*

On PanNuke, a **segmentation** error is an outline mistake (a nucleus missed, invented, or drawn the wrong size); a **class identification** error is the right nucleus with the wrong cell type.

| Test | Result |
|---|---|
| Wrong pixels that are segmentation errors | **69%** |
| Wrong pixels that are class identification errors | **31%** |
| True nuclei HoVer-Net finds | 80% |
| Found nuclei with the right type (voting → presence) | 80.7% → **82.8%** |
| Presence on HoVer-Net | 0.4541 → **0.4797** (+0.0256) |
| Upper bound: true types given per patch | **0.5262** (+0.0721) |
| Upper bound: every found nucleus the right type | **0.6902** (+0.2361) |

**Most wrong pixels are outlines, which WCSP can't touch**, but type mistakes cost most of the score. Presence recovers about a third of what a perfect presence rule could (+0.0256 of +0.0721).

### Results

#### This week's experiments

| Date | Experiment | Test set | Result (mIoU) |
|---|---|---|---|
| Nov 17 | Hierarchy decay: sibling, parent–child and fidelity schedules | 300 mixed-organ patches | All = raw HoVer-Net (0.3947, Δ ≈ 0); presence 0.4001 |
| **Nov 22** | **Breast: all 8 WCSP families fit and tuned on Breast only** | 169 Breast | **Only presence helps: 0.4344 → 0.4484 (+0.0141)** |
| Nov 22–23 | Colon: same protocol | 106 Colon | Presence +0.0019 (not significant) |
| Nov 22 | Adrenal gland: same protocol | 144 Adrenal | Presence +0.0026 (not significant) |
| **Nov 23** | **Wider presence sweep: 1,584 settings** | 600 fresh patches | **0.4326 → 0.4450 (+0.0124)** |
| Nov 23 | Preliminary error analysis | 400 dev patches | 69% segmentation, 31% class identification |
| Nov 23 | BCSS-style presence on Breast | Breast fold 2 | 0.4484 → 0.4519 (exploratory) |

*Gains are vs. fixed-mask voting (one label per detected nucleus), PanNuke's matched control.*

#### Per organ (fit and tuned on one tissue type)

| Organ | Test patches | Voting | Presence | Gain [95% CI] | Works? |
|---|---:|---:|---:|---:|---|
| **Breast** | 169 | 0.4344 | 0.4484 | +0.0141 [+0.0078, +0.0213] | **Yes** (Holm p = 0.003) |
| Colon | 106 | 0.4267 | 0.4286 | +0.0019 [−0.0013, +0.0055] | No |
| Adrenal gland | 144 | 0.3404 | 0.3429 | +0.0026 [−0.0052, +0.0111] | No |

The other seven families move mIoU by less than ±0.004 on every organ. **Per-organ tuning only paid off on Breast**; on Breast, fitting on Breast alone beats fitting on all tissue types (0.4384 → 0.4484).

#### Wider presence sweep (600 fresh patches, folds 1 and 3)

| Method | mIoU | vs. voting [95% CI] |
|---|---:|---:|
| HoVer-Net, raw | 0.4337 | |
| HoVer-Net, voting | 0.4326 | — |
| Presence, original setting | 0.4369 | +0.0043 [+0.0020, +0.0072] |
| **Presence, swept (strength 12, per-organ counts)** | **0.4450** | **+0.0124 [+0.0021, +0.0233]** |

**Stronger presence helps**, mostly on Breast and Colon; the 16 smaller organs see no gain.

---

## BCSS

### Preliminary experiments

**Goal:** find out whether **segmentation** or **class identification** is the bigger problem for the U-Net, so we know what to fix first.

*Full-resolution U-Net · 185 BCSS dev tiles · no training*

#### The two errors

- **Segmentation error:** the U-Net knows what a region is, but gets its shape wrong: pieces break off, borders are off.
- **Class identification error:** the U-Net gets the shape roughly right, but calls the region the wrong tissue.

#### Method

Take each true region (one connected piece of one class in the ground truth) and check the U-Net's most common label inside it. **Right** → its wrong pixels are segmentation errors. **Wrong** → they are class identification errors.

![Method](figures_20261123/prelim_split_diagram.png)

#### Test 1 · How much of each class is wrong, and why?

![Test 1](figures_20261123/prelim_error_split.png)

**Class identification is slightly bigger on average** (12.2% vs 9.4% of a class's pixels). It dominates for "other" and stroma, while segmentation dominates for inflammatory, necrosis and tumor.

#### Test 2 · What would fixing each one be worth?

- **Perfect class identification:** keep the U-Net's regions; give each the class most of its true pixels have.
- **Perfect segmentation:** keep the true regions; give each the class the U-Net predicts for most of it.

![Test 2](figures_20261123/prelim_ladder.png)

**Both matter about equally:** perfect class identification +0.1186, perfect segmentation +0.1028. Presence (+0.0086) and smoothing (+0.0023) recover only a small fraction.

### Results

Two kinds of fix: **WCSP fixes** correct the U-Net's output afterwards; **U-Net training losses** change the model itself.

#### 1 · WCSP fixes (after the U-Net): presence and clusters

*Targets class identification. The U-Net is unchanged; WCSP corrects its output.*

1. From the training ground truth, count how often each **set of classes** appears together in a tile (e.g. tumor + stroma is common; necrosis + "other" is rare).
2. For a new tile, score every possible class set: the U-Net's confidence in the classes it would keep, plus a penalty for how rare that set is.
3. Keep the best set and remove the other classes; their pixels go to the U-Net's next-best allowed class. Then smooth with WCSP.

**Cluster** (optional add-on): SLIC groups the tile into about 256 patches of similar colour and prediction, and each patch is labelled as one unit before a final pixel pass.

| Method | Dev (185) | Val (2,124, held out) | Full-res U-Net, dev |
|---|---:|---:|---:|
| U-Net | 0.7067 | 0.6489 | 0.6524 |
| WCSP | 0.7090 (+0.0023) | 0.6506 (+0.0018) | 0.6547 (+0.0023) |
| WCSP + cluster | 0.7163 (+0.0095) | 0.6544 (+0.0055) | — |
| WCSP + presence | 0.7229 (+0.0162) | 0.6550 (+0.0061) | 0.6610 (+0.0086) |
| **WCSP + presence + cluster** | **0.7242 (+0.0175)** | **0.6564 (+0.0075)** | — |

![BCSS fixes](figures_20261123/bcss_fixes.png)

**Presence works but recovers little:** +0.0075 on held-out val, against about +0.12 possible from perfect class identification.

#### 2 · U-Net training losses (changing the model itself)

*Targets both errors at the source. No WCSP afterwards.*

Instead of correcting the output afterwards, add a term to the U-Net's training loss. Same recipe for every model (10 epochs, same seed); only the loss changes.

- **Segmentation loss (B):** for every true region in a training tile, compute the average prediction over its pixels and penalize pixels that stray from it. A region predicted as one coherent class costs nothing; a patchwork costs a lot, even when cross-entropy rates the two the same.
- **Presence loss (C):** from the prediction, estimate which classes the tile contains, and penalize mismatches with the true class set.

| Model | Training loss | Targets | Pilot (2 epochs) | Full (10 epochs) |
|---|---|---|:---:|:---:|
| A · baseline | Pixel cross-entropy | — | running | 0.6679 |
| B | + segmentation loss | Segmentation | running | — |
| C | + presence loss | Class identification | running | — |

**Success** = B or C beats A on raw U-Net output.

#### Settings (from sweeps on dev)

| Setting | Tried | Used |
|---|---|---|
| WCSP strength λ | 0.5 – 32 | 8 |
| Presence strength γ | 10 – 100,000 | 3,000 |
| SLIC patches | 64 / 256 / 1,024 | 256 |
| Multi-scale μ, block levels, decay (23 schedules) | various | no gain over WCSP; not used |

---

<!-- BEGIN FULL512 BCSS -->
## BCSS full-resolution correction experiments

**Current finding:** connected-region moves and tile-level class context give
the strongest completed development results. These new choices have **not**
yet been evaluated on the full validation set.

### Data and comparison protocol

- Official upstream BCSS preparation v2: **151 ROIs**, 0.25 MPP, nonoverlapping
  **512×512** tiles; **7,768 fit / 1,012 dev / 4,028 validation** tiles.
- The full-resolution head and backbone were already trained. Every correction
  arm reuses the **same frozen checkpoint and probability maps** and scores the
  entire 512×512 field. No central crop or new U-Net training is used here.
- Correction selection uses **185 frozen dev tiles from 14 ROIs**, a fixed
  subset of the 1,012 dev tiles. The U-Net baseline on this subset is
  **0.652414 mIoU**. Do not compare it directly with the recap's 0.7067.
- Priors and the independent appearance model use the **fit split only**.
  Aggregate accuracy is reported only after all 185 tiles of an arm finish.
  Paired intervals resample whole ROIs (2,000 replicates, seed 0).
- Snapshot: **34/40 correction configurations complete**, 6,323/7,400 tile
  solves, November 23 at 15:16 PST. The remaining appearance configurations
  are still running. The earlier expanded sweep finished all 3,145 solves.

The earlier 0.6610 presence pilot used the existing prior and an
area-scaled fixed gamma. The following sweep refits a full-field fit prior,
varies gamma on dev, and uses audited convergence with matched controls.

### What the new methods change

| Family | Change | Configurations |
|---|---|---:|
| Temperature | Soften U-Net probabilities before constructing costs; makes class changes cheaper | T={1,1.5,2,3,4}, mu={0,1}: **10** |
| Block fidelity | Reduce the coarse blocks' own pooled U-Net costs; retain their neighbor and parent–child links | block_alpha={0,0.25,0.5}, mu=1: **3**; alpha=1 control shared |
| Class presence | Choose an allowed class set using fit-split co-occurrence costs plus pixel evidence, then spatially optimize | gamma={40,120,400,1200,4000,12000}, mu={0,1}: **12** |
| Region moves | Propose changing a connected same-label region or an image-derived SLIC segment together | Two region types, mu={0,1}: **4** |
| Image-aware boundaries | Add disagreement costs between pixels with similar HED stain features | beta={1,4}, mu={0,1}: **4** |
| Independent regional evidence | Blend U-Net costs with a fit-only HED/texture classifier pooled within SLIC segments | weights={0.1,0.3,1}, mu={0,1}: **6**, plus **1** appearance-only control |

All 40 arms use lambda=8; mu=0 disables parent–child coupling and mu=1
enables it. Temperature and appearance also change the evidence reconstructed
for coarse blocks. Block-alpha isolates their direct fidelity weight.

**Region moves:** first reach an ordinary single-variable local minimum.
Then test up to two passes over the 32 largest eligible regions (at least
64 pixels). Try each tissue label for the entire region, alone or together
with coarse blocks at least half covered by it. Accept a proposal only when
the **exact full objective decreases**, then resume pixel/coarse descent.
The final assignment is audited for single-variable stationarity; the bounded
proposal search does not establish a global or region-wise optimum. This is
different from the recap's SLIC cluster solver: the strongest new result uses
**connected components of the current labels**, not SLIC.

**Presence:** the full-field prior uses 6,888 fit tiles with at least 1,000
labeled pixels; a class is present when it occupies at least 1% of labeled
area. All 31 nonempty class sets are considered with add-one smoothing.
Selection is a two-stage approximation: choose an allowed set from unary
evidence and the prior, then impose a finite 1,000 exclusion cost on other
labels. Every result checks that excluded labels do not survive. This does
not jointly optimize class-set selection with the spatial energy.

**Appearance:** a diagonal Gaussian classifier uses HED channels, local
means and local standard deviations. Of 288 fit tiles, 256 fit its parameters
and 32 calibrate its temperature. Development RGB and annotations must align
exactly with the frozen inputs; dev annotations score results but do not
construct regions or appearance evidence. This is an independent shallow
appearance model, not another U-Net training arm.

### Completed development results

| Method | mIoU | Change vs. U-Net |
|---|---:|---:|
| Raw full-resolution U-Net | 0.652414 | — |
| Ordinary pixel WCSP, mu=0 | 0.654691 | +0.002276 |
| Ordinary hierarchy, mu=1 | 0.657038 | +0.004624 |
| Best expanded lambda/mu grid: lambda=16, mu=2 | 0.657713 | +0.005299 |
| Temperature T=4 + hierarchy | 0.658183 | +0.005768 |
| No block fidelity + hierarchy | 0.658301 | +0.005886 |
| **Presence gamma=4000 + hierarchy** | **0.664791** | **+0.012376** |
| **Connected-region moves, mu=0** | **0.667026** | **+0.014611** |
| HED boundary beta=1 + hierarchy | 0.656998 | +0.004584 |
| Appearance weight=0.1, mu=0; family incomplete | 0.655015 | +0.002600 |

![Full-512 development gains and matched comparisons](figures_20261123/bcss_full512_dev.png)

| Comparison | Change in mIoU | 95% paired ROI interval |
|---|---:|---:|
| Region moves minus ordinary pixel WCSP | +0.012335 | [+0.002673, +0.022119] |
| Presence + hierarchy minus ordinary hierarchy | +0.007752 | [+0.003623, +0.012651] |
| T=4 minus T=1, both with hierarchy | +0.001144 | [+0.000158, +0.002339] |
| No block fidelity minus ordinary block fidelity | +0.001262 | [−0.000969, +0.003266] |

These are **development-selected, exploratory intervals**, without correction
for selecting settings across the sweep. Removing block fidelity improves
the point estimate but its interval includes zero. HED boundary terms did
not consistently improve the hierarchy. The appearance grid is incomplete.

The convergence experiment adds a small gain: mu=1 reaches **0.657038** at
convergence versus **0.656822** after four outer rounds. More iterations
alone do not explain the larger region-move or presence gains.

### Corrections, failures and region interiors

- Region moves fix **601,023** previously wrong pixels and break **404,681**
  previously correct pixels, relative to the raw U-Net. The matched pixel
  control fixes 154,373 and breaks 112,729. A separate energy audit found
  lower pixel energy on 154 tiles, ties on 31, and no increases; 780 region
  proposals were accepted across the cohort. See the
  [per-tile energy audit](../experiments/bcss_wcsp/results/full512_20261123/region_energy_audit.json).
- Presence + hierarchy fixes **425,280** and breaks **231,814** pixels versus
  the raw U-Net. More than 10 pixels from a raw prediction boundary, it fixes
  **34,715** and breaks **3,170** pixels. The ordinary hierarchy fixes 15 and
  breaks 11 in those interiors.
- Region moves also change interiors, but their interior fixes and breaks
  are close (76,898 versus 76,175). Their benefit cannot be summarized as
  uniformly fixing large regions. Report per-class IoU and both fixes and
  breaks alongside pooled mIoU; energy reduction alone is not correctness.

![Representative region-move improvement and failure](figures_20261123/bcss_full512_region_examples.png)

Examples are the median positive and median negative **per-tile mIoU change
against the matched pixel control**, selected after inspecting dev outcomes.
They illustrate behavior, not independent evidence. The edit map is relative
to that pixel control: green=fix, red=break, orange=wrong-to-wrong. Ignored
annotation pixels are gray. Exact IDs and selection counts are in
[region_examples.json](../experiments/bcss_wcsp/results/full512_20261123/region_examples.json).

### Established full-validation result

The earlier frozen validation protocol accounts for **all 4,028 tiles from
45 held-out ROIs**, scoring 3,229 with labeled pixels and excluding 799 wholly
ignored tiles. It uses the trained full-512 checkpoint, lambda=8, T=1 and the
dev-selected mu=1 setting with the historical four-round limits.

| Full-validation method | mIoU |
|---|---:|
| Raw full-resolution U-Net | 0.670665 |
| Pixel WCSP, mu=0 | 0.672414 |
| **Multiscale WCSP, mu=1** | **0.673607** |
| New presence / region-move candidates | Not evaluated on full validation yet |

![Full-512 validation](figures_20261123/bcss_full512_validation.png)

The established improvement over the U-Net is **+0.002942 mIoU**, with
95% ROI interval **[+0.002218, +0.003615]**. The 0.667026 new dev score and
0.673607 validation score describe different populations and must not be
ranked against one another.

**Next:** finish the appearance grid, freeze the final development choices,
and evaluate the region-move and presence candidates with their matched
controls on the full validation set. Do not select settings using those
validation outcomes.

Portable evidence: [correction snapshot](../experiments/bcss_wcsp/results/full512_20261123/corrections_dev.json),
[completed expanded sweep](../experiments/bcss_wcsp/results/full512_20261123/expanded_dev.json),
[validation summary](../experiments/bcss_wcsp/results/full512_20261123/validation.json),
and [frozen provenance](../experiments/bcss_wcsp/results/full512_20261123/provenance.json).
Raw images, probability caches and per-tile predictions remain local.
Rebuild the charts with `python deliverables/figures_20261123/make_full512_figures.py`;
the qualitative figure additionally needs `--cache-root` pointing to the
original experiment checkout. The frozen experiment source hashes predate
the upstream solver updates; existing output roots must retain those sources.
Sync this section and its figures into the GitHub Pages version with
`python deliverables/figures_20261123/sync_full512_pages.py` (requires mistune).

The training-loss arms in the preceding BCSS section use the original
central-256 architecture/field, separately from this full-512 post-processing
protocol (`experiments/bcss_wcsp/slurm/train_loss_arms.sbatch`).
<!-- END FULL512 BCSS -->

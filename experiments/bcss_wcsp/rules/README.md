# BCSS: data-derived rules for WCSP

**Result.** One rule, **class presence**, gives almost all of the gain: dev mIoU 0.7067 (U-Net) → 0.7229 (+1.6 points). With cluster WCSP underneath it reaches **0.7242 (+1.75)**. Neighbor smoothing alone (the original WCSP) gives +0.2, and no other rule adds anything once presence is on.

Setup throughout: pretrained TIAToolbox U-Net, half resolution, central 256 × 256 crop, data-derived PMI table b, λ = 8. Dev = the 185 tiles every sweep uses (`--limit 200`). Rules come from the **fit** split's ground truth (4,588 tiles). All numbers are dev; the best setting still needs one frozen val run.

## 1. The class presence rule

**Idea:** before labeling pixels, decide **which classes this tile is allowed to contain**, then run WCSP with only those classes.

**Step 1: enumerate class sets.** With 5 classes there are 31 non-empty sets ({Tumor}, {Tumor, Stroma}, …).

**Step 2: score each set S with two costs:**

    cost(S) = Σ_pixels min_{k ∈ S} U_p(k)   +   γ · (−log P(S))
              └──── U-Net evidence ────┘       └── rarity ──┘

- **U-Net evidence.** U_p(k) = −log p(k) from the U-Net. If S leaves a class out, its pixels must take their best class within S and pay that class's −log p. Dropping a class the U-Net is confident about over a large area is expensive; dropping a few uncertain pixels is cheap.
- **Rarity.** P(S) = how often exactly this set of classes (each ≥ 1% of the tile) occurs in fit ground-truth tiles. γ = 3000 sets the trade-off (tuned on dev: 1000 → 0.7139, 3000 → 0.7229, 10,000 → 0.7094).

**Step 3:** pick the set with the lowest cost. All 31 sets are scored, so the choice is exact. Classes outside it are forbidden.

**Step 4:** run the normal WCSP with only the allowed classes; it relabels the forbidden pixels and cleans up borders.

**Which class gets dropped?** Not simply the smaller one: the one whose removal costs the least U-Net evidence relative to how much rarer the set with it is. In practice that is usually a small, low-confidence region. A large, confident region survives even in a rare combination.

Example: the U-Net labels a tile 60% tumor, 38% stroma and 2% scattered, low-confidence "other". Keeping "other" means the set {Tumor, Stroma, Other} (P = 0.020, cost 3.93) instead of {Tumor, Stroma} (P = 0.271, cost 1.31). That is an extra 2.6 × 3000 ≈ 7,900, more than the evidence those pixels carry, so "other" is dropped.

### The set cost table (fit ground truth, `configs/tables/bcss_presence_prior.json`)

| Classes in tile | P | −log P |
|---|---:|---:|
| Tumor + Stroma | 0.271 | 1.31 |
| Tumor | 0.184 | 1.70 |
| Inflammatory + Stroma | 0.127 | 2.06 |
| Stroma | 0.083 | 2.49 |
| Tumor + Inflammatory + Stroma | 0.074 | 2.60 |
| Tumor + Stroma + Necrosis | 0.046 | 3.09 |
| Necrosis | 0.039 | 3.24 |
| Inflammatory + Stroma + Other | 0.033 | 3.41 |
| Stroma + Other | 0.031 | 3.47 |
| … | | |
| Tumor + Stroma + Other | 0.020 | 3.93 |
| Stroma + Necrosis | 0.010 | 4.57 |
| Tumor + Other | 0.003 | 5.87 |
| Inflammatory + Necrosis | 0.0004 | 7.74 |
| Necrosis + Other | 0.0004 | 7.74 |
| All five classes (and 4 other unseen sets) | 0.0002 | 8.44 (floor) |

Unseen sets get a small floor probability, so nothing is strictly forbidden, only expensive.

### Why it works when the pair table did not

- **The pair table only acts on borders.** A false "other" blob inside stroma pays only along its edge, while its U-Net support grows with its area, so blobs larger than a few pixels survive. WCSP smooths edges but cannot delete regions.
- **The false borders look normal.** False "other" mostly touches stroma, a common pair in the truth (83% of "other" regions touch stroma). What is wrong is that "other" exists in that tile at all, which only a tile-level rule can see.
- **The U-Net's main error is tile-level:** it predicts "other" in 20% of dev tiles (truth 11%) and necrosis in 15% (truth 11%).

### What changes (dev)

72 of 185 tiles lose a class: "other" in 42, inflammatory in 22, necrosis in 17 and tumor in 7. 1.3% of pixels change (162k of 12.1M), mostly other → stroma (82k), inflammatory → stroma (19k), necrosis → tumor (13k), other → tumor (12k), other → necrosis (11k) and necrosis → stroma (11k).

| Class IoU | WCSP | + presence | Δ |
|---|---:|---:|---:|
| Tumor | 0.814 | 0.816 | +0.2 |
| Inflammatory | 0.758 | 0.759 | +0.1 |
| Stroma | 0.713 | 0.720 | +0.7 |
| Necrosis | 0.652 | 0.666 | +1.3 |
| **Other** | 0.607 | **0.653** | **+4.6** |

"Other" precision rises from 0.71 to 0.78 while recall barely drops (0.81 → 0.80): the rule removes "other" the U-Net invented.

## 2. Every candidate rule, measured in the data first

`rule_stats.py` → `rule_stats.json`. A rule is worth encoding only if the truth follows it and the U-Net does not.

| Rule (how often it is broken) | Fit truth | Dev truth | U-Net | WCSP | Encode? |
|---|---:|---:|---:|---:|---|
| Tile has a rare class set (P < 1%) | 3% | 2% | 10% | 9% | Yes: presence |
| Tiles containing "other" | 11% | 11% | 20% | 20% | Yes: per-class cost |
| Tiles containing necrosis | 12% | 11% | 15% | 15% | Yes: per-class cost |
| "Other" present but < 2% of the tile | 22% | 21% | 43% | 30% | Yes: minimum share |
| Necrosis present but < 2% of the tile | 9% | 5% | 23% | 16% | Yes: minimum share |
| Necrosis pieces per tile | 1.1 | 1.3 | 5.2 | 1.8 | Yes: minimum size |
| Inflammatory region with < 50% stroma border | 4% | 4% | 19% | 10% | Yes: border rule |
| Necrosis region with < 10% tumor border | **51%** | **60%** | 36% | 35% | **No**: the truth breaks it more than the U-Net |
| ROI context: P(class \| no other ROI tile has it) | ≈ 0 | — | — | — | **No**: every ROI contains nearly every class |

"Necrosis borders tumor" is a standard literature rule, but in BCSS most necrosis regions mostly border stroma or nothing at all.

## 3. Each rule alone on WCSP (dev mIoU)

| Rule | Setting | mIoU | Δ vs. WCSP |
|---|---|---:|---:|
| None (WCSP) | | 0.7090 | — |
| **Class presence** | γ = 3000 | **0.7229** | **+1.39** |
| Cluster WCSP | 256 SLIC clusters + pixel refine | 0.7163 | +0.73 |
| Per-class cost, other | 1,000 / 3,000 / 10,000 | 0.7108 / 0.7130 / 0.7158 | up to +0.68 |
| Per-class cost, necrosis | 3,000 / 10,000 | 0.7116 / 0.7145 | up to +0.55 |
| Per-class cost, inflammatory | 3,000 | 0.7103 | +0.13 |
| Neighbor-tile presence | strength 1000 | 0.7129 | +0.39 |
| Minimum share, other | 2% / 5% | 0.7104 / 0.7119 | up to +0.29 |
| Minimum share, every class | 2% | 0.7114 | +0.24 |
| Minimum share, necrosis | 2% | 0.7094 | +0.04 |
| Necrosis minimum size | 256 / 1,024 / 4,096 px | 0.7089 / 0.7100 / 0.7109 | up to +0.19 |
| Inflammatory borders stroma | 30% / 50% | 0.7092 / 0.7097 | +0.07 |
| Minimum piece size, every class | 64 / 256 px | 0.7090 / 0.7089 | 0 |

## 4. Combinations (all with class presence, γ = 3000)

| Combination | mIoU | Δ vs. presence alone |
|---|---:|---:|
| **Cluster WCSP + presence** | **0.7242** | **+0.13** |
| Cluster WCSP + presence + region rules¹ | 0.7235 | +0.06 |
| Presence alone | 0.7229 | — |
| + necrosis minimum size 1,024 px | 0.7231 | +0.02 |
| + "other" ≥ 5% of tile | 0.7229 | 0 |
| + inflammatory borders stroma (50%) | 0.7228 | −0.01 |
| Cluster WCSP + presence + per-class costs² + region rules¹ | 0.7218 | −0.11 |
| + neighbor-tile presence (1000) | 0.7206 | −0.23 |
| + per-class cost on other, 3,000 / 10,000 | 0.7203 / 0.7159 | −0.26 / −0.70 |

¹ Necrosis ≥ 1,024 px, "other" ≥ 5% of tile, inflammatory borders stroma (50%). ² Other 3,000, necrosis 3,000, inflammatory 1,000.

Four runs were still pending when this was written: presence + per-class necrosis cost; presence + all per-class costs, with and without the region rules; presence + region rules.

**Why stacking stops helping:** every rule that helps alone removes false "other" or necrosis. Presence already does that, so extra rules start deleting real "other".

## 5. Other experiments on the same setup

| Experiment | Best dev mIoU | Verdict |
|---|---:|---|
| Multi-scale block WCSP (μ = 0.5, 3 levels) | 0.7108 | +0.2 over WCSP |
| Block-level decay (23 schedules, PanNuke design) | 0.7108 | No schedule beats the control |
| Potts table instead of the data-derived b | 0.7106 | Ties: the biological pair structure adds little |
| SLIC clusters: image / colour + probs / probs (64 / 256 / 1,024) | 0.7143 (image, 256) | ≈ pixel refine |

**Bug fixed.** skimage 0.26 rescales float input to [0, 1] before SLIC, so with `convert2lab=False` the spatial term dominated and SLIC returned a regular grid whatever the features. `wcsp/solve/clusters.py` now divides compactness by the feature range. The first cluster sweep (`c_*`) ran on grids; `c2_*` and later runs use real superpixels.

## Reproduce

```bash
# measure the rules
python experiments/bcss_wcsp/rules/rule_stats.py
# per-rule and combination sweeps (one array task per line)
S=experiments/bcss_wcsp
sbatch --clusters=discovery --array=0-12 $S/slurm/sweep.sbatch $S/configs/bcss_rules.yaml $S/configs/sweeps/bcss_rules2_dev.txt --split dev --limit 200
sbatch --clusters=discovery --array=0-9  $S/slurm/sweep.sbatch $S/configs/bcss_rules.yaml $S/configs/sweeps/bcss_combo_dev.txt --split dev --limit 200
sbatch --clusters=discovery --array=0-2  $S/slurm/sweep.sbatch $S/configs/bcss_rules_clusters.yaml $S/configs/sweeps/bcss_combo_clusters_dev.txt --split dev --limit 200
```

Code: `wcsp/solve/presence.py` (class presence, neighbor rule, per-class costs), `wcsp/solve/regions.py` (minimum size, border, minimum share), `wcsp/solve/clusters.py` (cluster WCSP). Tests: `tests/wcsp/test_rules_clusters.py`.

## Next

1. One frozen val run of cluster WCSP + presence.
2. Rerun on Ryan's full-resolution inputs.
3. A stronger pixel model: `experiments/bcss_wcsp/fm/train_fm.py` fine-tunes an open pathology foundation model (Midnight-12k or Phikon-v2) with a light decoder; presence then runs on its output.

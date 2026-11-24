# PanNuke WCSP per organ: Breast, Colon, Adrenal gland

**Question:** does any PanNuke WCSP family work better when its counts and settings come from one organ instead of all organs mixed?

**Answer:** only the type-set WCSP, and only on Breast. On Colon and Adrenal, nothing tuned for the organ beats its control. The other seven families do nothing on any organ.

| Organ (who ran it) | Data | Test patches | Voting | Type set tuned for organ | Gain [95% CI] | Works? (Holm, 8 families) |
|---|---|---:|---:|---:|---|---|
| **Breast** (Kashvi) | fold 2 | 169 | 0.4344 | **0.4484** | **+0.0141 [+0.0078, +0.0213]** | **Yes**, p = 0.003 |
| Colon (Aakanksha) | fold 2 | 106 | 0.4267 | 0.4286 | +0.0019 [−0.0013, +0.0055] | No, p = 1.0 |
| Adrenal gland (Abhishek) | folds 1–3 | 144 | 0.3404 | 0.3429 | +0.0026 [−0.0052, +0.0111] | No, p = 1.0 |

The original all-organ type-set setting, unchanged, gains +0.0040 on Breast and +0.0049 on Adrenal [+0.0016, +0.0089].

## Every family, gain over its matched control (test)

| Family | Breast | Colon | Adrenal |
|---|---:|---:|---:|
| Pixel learned table | +0.0001 | +0.0004 | −0.0005 |
| Hierarchy V1/V2 | −0.0000 | +0.0001 | +0.0000 |
| Scaling + siblings | −0.0001 | +0.0000 | −0.0001 |
| Decay schedules | −0.0000 | +0.0000 | −0.0000 |
| Nucleus pairwise | −0.0009 | +0.0004 | −0.0015 |
| Type counts | −0.0038 | −0.0008 | −0.0007 |
| **Type set** | **+0.0141** | +0.0019 | +0.0026 |
| Raw TP component | +0.0032 | +0.0018 | −0.0015 |

## Why Breast and not the others

- Breast has the most patches to count from, and its kinds of cells follow a strong pattern: tumor and normal epithelium almost never appear together. HoVer-Net breaks that pattern often, and the Breast-tuned type set fixes it.
- Colon and Adrenal have fewer patches, so the settings chosen on their dev sets didn't hold up on test.
- `../type_set_sweep_20261123/` confirms this on 600 fresh all-organ patches: per-organ counts help Breast and Colon but not the 16 smaller organs.

## Where the files are

| Organ | Location |
|---|---|
| Breast | `breast/`, branch `organ-wcsp-breast` |
| Colon | Aakanksha's run (`organ_wcsp/colon/`, not yet pushed). The numbers above are copied from her report of 23 November. |
| Adrenal gland | `adrenal_gland/` (this folder; README and DECISIONS.md there) |

All three use the same protocol: exclude the 1,235 previously used fold 2 IDs, split 40/20/40 into fit/dev/test with seed 20261122, fit tables on fit only, tune on dev, freeze, score test once, 5,000-replicate paired bootstrap, Holm across 8 families. Each organ used its own script, so small details differ (rounding of split sizes, size of the dev subset used for the slowest families). See each organ's README.

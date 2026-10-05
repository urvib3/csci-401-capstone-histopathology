# Deliverable 3: baselines and WCSP-ready representations

*Joint report for the PanNuke and BCSS benchmarks. Last updated 2026-10-04. Details, commands and per-person write-ups live in [`pannuke/README.md`](pannuke/README.md) and [`bcss/README.md`](bcss/README.md).*

## Summary

Deliverable 3 set up baselines for each dataset (one on PanNuke, three on BCSS), checked how good their outputs are, and looked at how to shrink those outputs into a small set of WCSP variables. No WCSP has been built or solved yet on either dataset; the numbers below are the baselines a WCSP step would have to improve on.

| | PanNuke | BCSS |
| --- | --- | --- |
| What is labeled | individual nuclei (5 cell classes) | tissue regions (5 coarse classes) |
| Baselines | HoVer-Net, pretrained PanNuke checkpoint (TIAToolbox) | (1) Phikon-v2 features + linear probe; (2) pretrained U-Net (TIAToolbox); (3) unsupervised Phikon-v2 codebook, clusters named by train labels |
| Data scored | Fold 2, all 2,523 patches | BCSS_512 val, 2,768 tiles from hospitals unseen in train |
| Headline result | detection F1 0.833, multi-class PQ 0.437 | val mIoU 0.652 (probe), 0.649 (U-Net), 0.561 (clusters, labels only for naming) |
| Natural WCSP variable | one detected nucleus (or a DBSCAN group of nuclei) | one region: a superpixel, a feature cluster or a grid cell |
| Variable reduction | DBSCAN: 55 → 47 variables on 10 patches (−15%) | Phikon-v2 per-tile clusters: ≈ 80 regions per 512 px tile at 0.79 oracle mIoU |

## PanNuke (nuclei)

**Data and baseline.** PanNuke Fold 2 (2,523 patches of 256 × 256 px). HoVer-Net was chosen because one model segments and classifies nuclei and a pretrained PanNuke checkpoint runs on a free Colab GPU in about a minute for the whole fold. It found 25,301 nuclei; 185 patches had none. *(Aakanksha)*

**Evaluation against ground truth** (central 164 × 164 px that HoVer-Net "fast" predicts; centroid matching within 12 px; PQ at IoU > 0.5, averaged over the 19 tissues). *(Abhishek)*

| Metric | Value |
| --- | ---: |
| Detection precision / recall / F1 | 0.881 / 0.790 / 0.833 |
| Class accuracy of detected nuclei | 0.823 |
| Detection + classification F1 | 0.686 |
| Binary PQ / multi-class PQ | 0.645 / 0.437 |

- 21% of real nuclei are missed and 18% of detected nuclei get the wrong class; dead cells are hardest (F1 0.347).
- Most common mix-ups: connective predicted as neoplastic (715), inflammatory and connective swapped both ways (468 / 464).
- Caveat: we could not confirm which folds the checkpoint was trained on, so these numbers may be optimistic.

**WCSP mapping and clustering** (10 sample patches, 55 nuclei). *(Kashvi)* Each nucleus becomes a variable over the 5 cell classes with unary cost −log p. Clustering nucleus centroids to cut the variable count:

| Method | Variables | Reduction | Groups mixing predicted classes |
| --- | ---: | ---: | ---: |
| None | 55 | 0% | 0 |
| K-means (K = ⌈N/3⌉) | 21 | 62% | 9 |
| DBSCAN (eps 40 px) | 47 | 15% | 0 |
| Spatial Ward (40 px graph) | 47 | 15% | 0 |

DBSCAN is the initial pick: same result as Ward with simpler settings, and isolated nuclei stay separate. HoVer-Net's CSV stores only the top class's pixel-vote share, so real −log p unaries need a re-run that keeps its full class probabilities.

## BCSS (tissue regions)

**Data.** BCSS_512 from Kaggle, 22 label codes merged to 5 classes (tumor, stroma, inflammatory, necrosis, other). Train 6,000 tiles / 106 ROIs, of which 14 ROIs (693 tiles) are held out as a dev split; val 2,768 tiles / 45 ROIs from 6 held-out hospitals. Class balance in train: 42% tumor, 35% stroma, 13% inflammatory, 8% necrosis, 3% other. *(Ryan)*

**Over-segmentation benchmark** (60 val tiles, native 512 px, Macenko-normalized). Ten unsupervised methods were swept over their size parameter and scored by how well their regions *could* be labeled (oracle mIoU = mIoU if each region takes its majority true class) and by boundary recall. Selected rows at comparable size:

| Method | Regions / tile | Oracle mIoU | Boundary recall | s / tile |
| --- | ---: | ---: | ---: | ---: |
| Phikon-v2 tokens, per-tile k-means (k = 16) | 79 | 0.787 | 0.34 | 0.1 + features |
| Quickshift | 408 | 0.882 | 0.43 | 5.5 |
| RAG greedy merge from SLIC | 384 | 0.834 | 0.51 | 2.4 |
| SLIC (Lab) | 161 | 0.811 | 0.37 | 0.5 |
| Felzenszwalb | 270 | 0.748 | 0.37 | 0.8 |
| k-means pixel clusters + components | 395 | 0.723 | 0.54 | 0.3 |
| Normalized cut | 40 | 0.574 | 0.17 | 5.8 |

**Tissue classification baselines** (all 2,768 val tiles):

| Method | Labels used | Val mIoU | Pixel acc |
| --- | --- | ---: | ---: |
| Phikon-v2 codebook: k-means k = 50 on frozen patch tokens, clusters named by majority train label | naming only | 0.561 | 0.775 |
| Phikon-v2 linear probe on frozen patch tokens | train tiles | 0.652 | 0.836 |
| ResNet-50 U-Net, pretrained BCSS checkpoint (TIAToolbox, no training by us) | fully supervised (by TIA) | 0.649 | 0.841 |

Phikon-v2 rows use 16 × 16 patch tokens per tile and score whole tiles at 256 px. The U-Net predicts only the central 256 px of each 512 px tile and is scored there at full resolution, so the comparison is close but not exact. We could not confirm which ROIs the U-Net checkpoint was trained on. Details in [`bcss/README.md`](bcss/README.md#3-pretrained-u-net-baseline-tiatoolbox).

Findings:

- Pathology foundation-model features are the strongest signal found so far: few regions per tile, good oracle accuracy, and 0.56 mIoU with labels used only to name clusters.
- Among colour-only methods, quickshift and RAG merging give the best regions; SLIC is the best speed/quality trade-off.
- Pixel colour clusters follow nuclei and stain edges, not tissue boundaries (highest boundary recall, lowest oracle accuracy).
- Hardest classes: "other" (IoU 0.43 for the probe) and stroma vs inflammatory (10% of stroma pixels predicted inflammatory).

## How the two datasets map to a WCSP

| | PanNuke | BCSS |
| --- | --- | --- |
| Variable | a nucleus or a group of nearby nuclei | a region (superpixel / feature cluster) or a 16 px grid cell |
| Domain | 5 cell classes | 5 tissue classes |
| Unary cost | −log p from HoVer-Net (needs full class probabilities) | −log p from the Phikon-v2 probe (saved per 32 px cell) |
| Binary cost | spatial neighbours within a radius | adjacent regions; class co-occurrence from `bcss.stats` |
| What a WCSP can fix | wrong classes of detected nuclei | wrong or noisy region labels, region boundaries |
| What it cannot fix | nuclei HoVer-Net never detected (21%) | errors inside a region if regions are too coarse |

## Open items

- [ ] Build and solve the first WCSP on each dataset and compare against these baselines
- [ ] PanNuke: commit the `results/evaluation/` CSVs the README links to (overall, per-class, confusion matrix, per-tissue)
- [ ] PanNuke: re-run HoVer-Net keeping full class probabilities for unary costs; consider CellViT as a second baseline
- [ ] BCSS: unsupervised pixel classification on stain colour ([handoff](bcss/HANDOFF_unsupervised_pixel_classification.md))
- [x] BCSS: pretrained supervised U-Net baseline (TIAToolbox), val mIoU 0.649
- [ ] BCSS: rescore the U-Net on whole tiles at 256 px so it is directly comparable with the probe
- [ ] BCSS: DINOv2 (natural-image) comparison and a U-Net trained from scratch (deferred; scripts ready)

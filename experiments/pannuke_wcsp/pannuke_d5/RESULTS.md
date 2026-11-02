# Fixed beta 0.30: learned pixel compatibility costs

Only raw TP argmax and frozen WCSP were evaluated. No sweep or post-test tuning.

| Metric | Raw TP | WCSP |
|---|---:|---:|
| Pooled biological mIoU | 0.376874 | 0.375209 |
| Pooled six-class mIoU | 0.473138 | 0.471753 |
| Pixel accuracy | 0.933362 | 0.933574 |

| Class | GT pixels | Test patch coverage | Raw TP IoU | WCSP IoU |
|---|---:|---:|---:|---:|
| Background | 456,622 | 20/20 | 0.954456 | 0.954476 |
| Neoplastic | 45,575 | 10/20 | 0.631903 | 0.633773 |
| Inflammatory | 4,212 | 8/20 | 0.160374 | 0.153589 |
| Connective | 18,105 | 15/20 | 0.412998 | 0.411050 |
| Dead | 1,301 | 3/20 | 0.155398 | 0.152301 |
| Non-Neoplastic Epithelial | 12,105 | 5/20 | 0.523697 | 0.525330 |

Biological mIoU difference: **-0.001666**. Paired patch-bootstrap 95% CI: **[-0.004971, +0.007969]**, 5,000 replicates; seed 20261102.
No reliable improvement is established: the interval includes zero. The observed biological mIoU decreases; uncertainty includes both decreases and increases.

The 114 net corrections improve pixel accuracy, but inflammatory, connective and Dead IoUs fall. Biological mIoU gives each class equal weight, so pixel-count gains do not guarantee a biological mIoU gain.

Changed: 1,820; wrong → correct: 801; correct → wrong: 687; wrong → different wrong: 332.
Background is 84.89% of GT. No GT class is absent across the pooled test set. Dead is rare (1,301 test pixels across three patches; 2,842 training pixels across two patches). Bootstrap replicates may omit rare-class patches, but fixed class denominators and the zero-union policy remain unchanged.

Inference: 17.039s; solver including audits: 1.066s; model setup: 931.782s. The explicit synthetic warmup and beta-zero validation are excluded; any additional real-layout JIT specialization inside a timed solve is included.

| Fold 2 patch | Tissue | Rounds | Stop | Verified local minimum | Residual gain | Fixes | Breaks | Inference s | Solver s |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 2324 | Stomach | 4 | no_changes | True | 0 | 35 | 33 | 1.020 | 0.213 |
| 0910 | Lung | 7 | no_changes | True | 0 | 53 | 88 | 0.928 | 0.051 |
| 2334 | Stomach | 5 | no_changes | True | 0 | 62 | 8 | 0.842 | 0.037 |
| 1603 | Colon | 2 | no_changes | True | 0 | 3 | 0 | 0.870 | 0.025 |
| 1113 | Adrenal_gland | 2 | no_changes | True | 0 | 5 | 4 | 0.857 | 0.021 |
| 0205 | Breast | 3 | no_changes | True | 0 | 13 | 8 | 0.823 | 0.031 |
| 0888 | Esophagus | 4 | no_changes | True | 0 | 12 | 2 | 0.840 | 0.033 |
| 2361 | Testis | 3 | no_changes | True | 0 | 14 | 10 | 0.803 | 0.028 |
| 2089 | Liver | 20 | no_changes | True | 0 | 83 | 97 | 0.832 | 0.142 |
| 1273 | Bile-duct | 2 | no_changes | True | 0 | 5 | 6 | 0.840 | 0.024 |
| 2029 | Kidney | 4 | no_changes | True | 0 | 23 | 36 | 0.854 | 0.035 |
| 1965 | HeadNeck | 3 | no_changes | True | 0 | 1 | 0 | 0.860 | 0.027 |
| 2174 | Pancreatic | 5 | no_changes | True | 0 | 31 | 31 | 0.837 | 0.038 |
| 2286 | Skin | 5 | no_changes | True | 0 | 49 | 25 | 0.857 | 0.038 |
| 0969 | Cervix | 13 | no_changes | True | 0 | 91 | 51 | 0.828 | 0.078 |
| 2238 | Prostate | 4 | no_changes | True | 0 | 19 | 28 | 0.824 | 0.034 |
| 2143 | Ovarian | 7 | no_changes | True | 0 | 69 | 58 | 0.833 | 0.054 |
| 2424 | Thyroid | 8 | no_changes | True | 0 | 46 | 55 | 0.831 | 0.055 |
| 1497 | Bladder | 7 | no_changes | True | 0 | 64 | 54 | 0.812 | 0.051 |
| 0930 | Uterus | 6 | no_changes | True | 0 | 123 | 93 | 0.847 | 0.052 |

All beta-zero outputs exactly match raw TP argmax. Every round has an independently recomputed non-increasing energy; local-move checks use separate vectorized code and tolerance 1e-10. No global optimum is claimed.

The 200 training and 20 test patches are disjoint, across 18 and 19 tissues respectively. Prior-use scan excludes 266 patch IDs, but unrecorded use cannot be verified. No slide/patient identifiers are available. Held out refers to WCSP fitting; pretrained HoVer-Net Fold 2 exposure is unknown.

Pairwise preferences cannot enforce literal stromal barriers or detect dense immune clusters. The learned table sees only two Dead-containing training patches; rare-class cost estimates need caution. Test selection intentionally enriches coverage and is not a prevalence-representative sample.

Checkpoint SHA256: `789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8`.
Frozen config SHA256: `444c64b66b5724fe1451a9bf6c355c735bd920d9ab2bb54a3759950d44e0573a`.
Starting commit: `95584cf12c828f371694c2b0103321dfa96625b1`. No commit or push made.

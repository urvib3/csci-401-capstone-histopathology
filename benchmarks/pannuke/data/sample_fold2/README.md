# PanNuke Fold 2 sample

This directory contains the first ten patches from the official Fold 2 archive, matching `image_000.png` through `image_009.png` in the prediction CSV.

- `images.npy`: shape `(10, 256, 256, 3)`, RGB tissue images, original float64 values.
- `masks.npy`: shape `(10, 256, 256, 6)`, original float64 annotations. Channels 0 through 4 contain nucleus instance IDs for neoplastic, inflammatory, connective, dead, and non-neoplastic epithelial cells, respectively. Zero means no nucleus of that class at the pixel. Channel 5 contains background.
- Access a class annotation with `masks[patch_index, y, x, class_channel]`. Positive values identify individual annotated nuclei within that class and patch.

Source: [University of Warwick PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/), Fold 2. The complete dataset is available there; only the ten-patch sample used by this analysis is included here.

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Citation: Gamper et al., *PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification* (2019); Gamper et al., *PanNuke Dataset Extension, Insights and Baselines* (2020), [paper](https://arxiv.org/abs/2003.10778).

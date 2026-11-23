# Type-set sweep: decision log

All on 2026-11-23.

| # | Decision | Why |
|---|---|---|
| 1 | Sweep only the type-set WCSP. | It's the only PanNuke cost that has ever helped, and it's PanNuke's version of BCSS's class-presence rule, which BCSS asked us to mirror. |
| 2 | Use fresh images from PanNuke folds 1 and 3, leaving out Adrenal_gland. | Re-tuning on images already used for a final score would be cheating. Folds 1 and 3 were untouched except Adrenal, which the 22 November Adrenal run used completely. |
| 3 | Split: 600 final-score, 400 knob-picking, all remaining 4,089 for counting. One shuffle, seed 20261123. | Counting needs only the true labels, not HoVer-Net, so it can use many images for free. 600 final-score images give intervals about half as wide as the per-organ runs. HoVer-Net on 1,000 images takes about an hour. |
| 4 | Knobs: strength 0.25–32, three per-nucleus costs, evidence penalty on/off, three group schemes, four smoothing weights, per-organ on/off (1,584 settings). | Strength: Breast and Adrenal both chose 4, the top of the old range. The others are every remaining choice in the original code, plus per-organ counting from this week's work. |
| 5 | Per-organ counts are smoothed toward the all-organ counts with the same smoothing weight. | Small organs have few counting images. Without this, an unseen combination in a small organ would be treated as nearly impossible. |
| 6 | Winner = best knob-picking mIoU. Within 0.0001, prefer the setting closest to the original and weaker. | Same rule as the earlier experiments. Only one setting fell within 0.0001, so the tie rule wasn't used. |
| 7 | Two primary comparisons, Holm-adjusted: winner vs. voting, and winner vs. the original setting with its original counts. Everything else is exploratory. | Written in `protocol.json` before any scoring. |
| 8 | Also report the original setting recounted on the 4,089 images. | Separates "more counting data" from "better knobs". |
| 9 | Checked that this script's counting reproduces `presence_20261114/priors.json` exactly, and that its solver matches the original `predict()` on all 400 knob-picking images. | Guarantees the sweep runs the same cost as the original, not a reimplementation with subtle differences. |
| 10 | After freezing, scored the winner at every strength on the final images and published it, labeled exploratory. | Answers "is stronger better?" directly. It was not used to choose anything. |
| 11 | Headline: the winner beats voting; beating the original setting is not established. | Holm p = 0.034 for the first and 0.10 for the second. |

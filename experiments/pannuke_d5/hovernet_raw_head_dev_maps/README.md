# Original pixel probability check

Same 15 development patches, checkpoint and central 164 × 164 crop as the previous experiments. No WCSP, clustering, nucleus voting or ground-truth input to prediction. Ground truth is used only for evaluation.

| Readout | Mean biological mIoU | Mean nucleus IoU | Mean nucleus Dice |
|---|---:|---:|---:|
| Saved postprocessed HoVer-Net | 0.474933 | 0.746156 | 0.851906 |
| Original TP softmax, six-channel pixel argmax | 0.438640 | 0.732753 | 0.842571 |
| Combined NP × conditional TP, pixel argmax | 0.443796 | 0.754598 | 0.857268 |

mIoU is averaged over biological classes present in each ground-truth patch, then averaged across patches. The raw TP readout improves one patch and worsens fourteen relative to saved HoVer-Net. These are development results, not a held-out test.

The original six-channel probabilities are retained as `type_probability_all` in each NPZ. This is softmax of all original TP logits before TIAToolbox's argmax. `nucleus_probability` separately preserves the NP foreground probability. The earlier combined readout is a constructed fusion of the two heads, rather than the original TP output.

Direct raw TP argmax does not improve this baseline. This check establishes access to the original pixel information; it does not test WCSP using those probabilities or establish the professor's intended formulation. The saved postprocessed baseline uses instance processing, so this comparison also changes postprocessing.

See `comparison.json` for per-patch/per-class metrics and `pixel_head_comparison.png` for a visual example. Reproduce evaluation with:

```sh
python3 experiments/pannuke_d5/compare_probability_heads.py --maps experiments/pannuke_d5/hovernet_raw_head_dev_maps --reference experiments/pannuke_d5/two_stage_dev_trial
```

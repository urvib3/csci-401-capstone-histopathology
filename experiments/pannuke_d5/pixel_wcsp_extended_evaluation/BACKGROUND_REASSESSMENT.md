# Metrics re-evaluated including background

Re-evaluation of the same 100 saved prediction pairs. No optimization, label changes, parameter tuning or per-patch ground-truth acceptance gate was performed.

| Method | Pooled six-class mIoU including background | Mean patch mIoU including background (GT-present classes) | Pixel accuracy |
|---|---:|---:|---:|
| hovernet | 0.608028 | 0.723194 | 0.937902 |
| wcsp | 0.636522 | 0.723400 | 0.938280 |

Accumulate pixel confusion matrix across patches, then average IoU equally across all six labels including background. Any label with zero dataset union is undefined and omitted.

For each patch average IoU over its GT-present classes including background, using the same class set for both methods; then average patches. Retained for comparison with earlier patch-mean metrics. Classes absent in GT are not separately averaged by this diagnostic.

## Per-class pooled IoU

| Class | HoVer-Net | WCSP |
|---|---:|---:|
| Background | 0.945354 | 0.944390 |
| Neoplastic | 0.744460 | 0.747618 |
| Inflammatory | 0.559568 | 0.565908 |
| Connective | 0.517862 | 0.516767 |
| Dead | 0.330962 | 0.493309 |
| Non-neoplastic epithelial | 0.549966 | 0.551143 |

Using the background-inclusive patch metric: 45 patches improve, 55 worsen and 0 are unchanged.

The earlier headline metric averaged only GT-present biological classes. Background was already a possible WCSP label and background errors already affected biological unions and pixel accuracy; its own IoU was excluded from that headline average. The new pooled six-class metric also changes the aggregation from a patch mean to dataset-pooled intersections/unions. Both changes are shown explicitly; the new pooled result must not be compared directly to the old patch mean as if the model improved. A dataset-level mean gain does not establish non-degradation for every patch.

Low WCSP energy guarantees neither higher IoU nor higher ground-truth accuracy. Rejecting changes based on each evaluation patch's ground truth would leak evaluation labels. The unchanged HoVer-Net baseline is reported separately from actual WCSP results.

[Full metrics](background_reassessment.json) · [Paired patch metrics](background_reassessment_per_patch.csv)

## Interpretation on these 100 patches

The background-inclusive patch mean changes only 0.723194 → 0.723400; 55 patches still worsen. The pooled six-class mean rises 0.608028 → 0.636522, but 95.0% of that net gain comes from the Dead-class IoU increase. Only four patches contain GT Dead pixels. This is not evidence that adding background fixed the optimization or that every patch is protected.

Missed nucleus pixels increase 59,171 → 64,943; false-positive nucleus pixels decrease 64,880 → 61,617. Background IoU itself decreases. The objective still needs review rather than claiming a non-degradation guarantee from these averages.

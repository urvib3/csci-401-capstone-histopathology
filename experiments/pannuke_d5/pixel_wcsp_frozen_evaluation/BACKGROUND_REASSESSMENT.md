# Metrics re-evaluated including background

Re-evaluation of the same 20 saved prediction pairs. No optimization, label changes, parameter tuning or per-patch ground-truth acceptance gate was performed.

| Method | Pooled six-class mIoU including background | Mean patch mIoU including background (GT-present classes) | Pixel accuracy |
|---|---:|---:|---:|
| hovernet | 0.564511 | 0.688440 | 0.927387 |
| wcsp | 0.574582 | 0.691650 | 0.928646 |

Accumulate pixel confusion matrix across patches, then average IoU equally across all six labels including background. Any label with zero dataset union is undefined and omitted.

For each patch average IoU over its GT-present classes including background, using the same class set for both methods; then average patches. Retained for comparison with earlier patch-mean metrics. Classes absent in GT are not separately averaged by this diagnostic.

## Per-class pooled IoU

| Class | HoVer-Net | WCSP |
|---|---:|---:|
| Background | 0.944308 | 0.942979 |
| Neoplastic | 0.587220 | 0.599756 |
| Inflammatory | 0.548754 | 0.567705 |
| Connective | 0.464061 | 0.463091 |
| Dead | 0.231379 | 0.262258 |
| Non-neoplastic epithelial | 0.611346 | 0.611705 |

Using the background-inclusive patch metric: 13 patches improve, 7 worsen and 0 are unchanged.

The earlier headline metric averaged only GT-present biological classes. Background was already a possible WCSP label and background errors already affected biological unions and pixel accuracy; its own IoU was excluded from that headline average. The new pooled six-class metric also changes the aggregation from a patch mean to dataset-pooled intersections/unions. Both changes are shown explicitly; the new pooled result must not be compared directly to the old patch mean as if the model improved. A dataset-level mean gain does not establish non-degradation for every patch.

Low WCSP energy guarantees neither higher IoU nor higher ground-truth accuracy. Rejecting changes based on each evaluation patch's ground truth would leak evaluation labels. The unchanged HoVer-Net baseline is reported separately from actual WCSP results.

[Full metrics](background_reassessment.json) · [Paired patch metrics](background_reassessment_per_patch.csv)

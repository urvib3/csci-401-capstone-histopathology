"""Evaluate the full Fold 2 HoVer-Net predictions against PanNuke ground truth.

Detection and classification use one-to-one centroid matching. Panoptic
Quality uses one-to-one mask matching at IoU > 0.5.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

BASE = Path(__file__).resolve().parents[1]
PREDICTION_DIR = BASE / 'results/sample_predictions/fold2'
OUTPUT_DIR = BASE / 'results/evaluation'
CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Non-neoplastic epithelial']
UNTYPED = 'Background (untyped)'
FOLD_SIZE = 2523
INPUT_SIZE = 256
OUTPUT_SIZE = 164
OFFSET = (INPUT_SIZE - OUTPUT_SIZE) // 2
MATCH_RADIUS = 12
MATCH_IOU = 0.5


def load_predictions():
    maps, names = [], []
    for path in sorted(PREDICTION_DIR.glob('hovernet_instance_maps_*.npz')):
        batch = np.load(path)
        maps.append(batch['predictions'])
        names.extend(batch['image_names'].tolist())
    maps = np.concatenate(maps)
    if maps.shape != (FOLD_SIZE, OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError('Expected one 164x164 instance map per Fold 2 image.')
    # Batches are saved in image order; the number in each filename is the Fold 2 index.
    if [int(Path(name).stem.split('_')[1]) for name in names] != list(range(FOLD_SIZE)):
        raise ValueError('Instance maps are not in Fold 2 order.')
    table = pd.read_csv(PREDICTION_DIR / 'hovernet_fold2_predictions.csv')
    if table.duplicated(['image_index', 'nucleus_id']).any():
        raise ValueError('Nucleus IDs must be unique within each patch.')
    if not table.predicted_type_id.between(0, 5).all():
        raise ValueError('Expected model class IDs 0 through 5.')
    return maps, table


def ground_truth_instances(mask):
    """Flatten the five class channels into one instance map and a class ID per instance."""
    instance_map = np.zeros(mask.shape[:2], dtype=np.int32)
    classes = []
    for channel in range(len(CLASS_NAMES)):
        class_mask = mask[..., channel]
        for instance in np.unique(class_mask[class_mask > 0]):
            classes.append(channel + 1)
            instance_map[class_mask == instance] = len(classes)
    # A nucleus can be fully overwritten where class channels overlap.
    kept = np.unique(instance_map[instance_map > 0])
    relabel = np.zeros(len(classes) + 1, dtype=np.int32)
    relabel[kept] = np.arange(1, len(kept) + 1)
    return relabel[instance_map], np.array(classes, dtype=int)[kept - 1]


def centroids(instance_map, count):
    ys, xs = np.nonzero(instance_map)
    labels = instance_map[ys, xs]
    areas = np.bincount(labels, minlength=count + 1)[1:]
    x = np.bincount(labels, weights=xs, minlength=count + 1)[1:] / areas
    y = np.bincount(labels, weights=ys, minlength=count + 1)[1:] / areas
    return np.column_stack([x, y])


def pairwise_iou(truth_map, prediction_map, truth_count, prediction_count):
    width = prediction_count + 1
    overlap = np.bincount(
        truth_map.ravel().astype(np.int64) * width + prediction_map.ravel(),
        minlength=(truth_count + 1) * width,
    ).reshape(truth_count + 1, width)
    truth_area = overlap.sum(axis=1)[1:, None]
    prediction_area = overlap.sum(axis=0)[None, 1:]
    overlap = overlap[1:, 1:]
    return overlap / (truth_area + prediction_area - overlap)


def match_centroids(truth_points, prediction_points):
    if not len(truth_points) or not len(prediction_points):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    distance = cdist(truth_points, prediction_points)
    # Out-of-range pairs share one large cost, so in-range pairs are maximized first.
    rows, columns = linear_sum_assignment(np.where(distance <= MATCH_RADIUS, distance, 1e6))
    valid = distance[rows, columns] <= MATCH_RADIUS
    return rows[valid], columns[valid]


def panoptic_counts(iou, truth_selected, prediction_selected):
    """Return TP, FP, FN, and summed IoU. IoU > 0.5 makes each match unique."""
    matched = iou[np.ix_(truth_selected, prediction_selected)]
    matched = matched[matched > MATCH_IOU]
    true_positive = len(matched)
    return np.array([
        true_positive,
        prediction_selected.sum() - true_positive,
        truth_selected.sum() - true_positive,
        matched.sum(),
    ])


def panoptic_quality(counts):
    true_positive, false_positive, false_negative, iou_sum = counts
    if true_positive == 0:
        return 0.0, 0.0, 0.0
    detection = true_positive / (true_positive + 0.5 * false_positive + 0.5 * false_negative)
    segmentation = iou_sum / true_positive
    return detection, segmentation, detection * segmentation


def scores(true_positive, predicted, truth):
    precision = true_positive / predicted if predicted else np.nan
    recall = true_positive / truth if truth else np.nan
    f1 = 2 * true_positive / (predicted + truth) if predicted + truth else np.nan
    return precision, recall, f1


def evaluate(masks, tissues, prediction_maps, table):
    class_count = len(CLASS_NAMES)
    # Rows: ground-truth class 1-5, then no ground truth. Columns: untyped 0, class 1-5, then missed.
    confusion = np.zeros((class_count + 1, class_count + 2), dtype=int)
    mask_counts = np.zeros(4)
    class_mask_counts = np.zeros((class_count, 4))
    image_rows = []
    predicted_types = {index: group.sort_values('nucleus_id') for index, group in table.groupby('image_index')}
    for index in range(FOLD_SIZE):
        crop = masks[index, OFFSET:OFFSET + OUTPUT_SIZE, OFFSET:OFFSET + OUTPUT_SIZE]
        truth_map, truth_classes = ground_truth_instances(np.asarray(crop))
        # The k-th smallest map label is CSV nucleus_id k; a few maps skip label 1.
        labels, prediction_map = np.unique(prediction_maps[index], return_inverse=True)
        prediction_map = prediction_map.reshape(OUTPUT_SIZE, OUTPUT_SIZE)
        if labels[0] != 0:
            raise ValueError(f'Instance map for image {index} has no background.')
        prediction_count = len(labels) - 1
        rows = predicted_types.get(index)
        prediction_classes = np.empty(0, dtype=int) if rows is None else rows.predicted_type_id.to_numpy()
        if len(prediction_classes) != prediction_count:
            raise ValueError(f'Instance map and CSV disagree for image {index}.')
        truth_points = centroids(truth_map, len(truth_classes))
        prediction_points = centroids(prediction_map, prediction_count)
        if prediction_count and not np.allclose(prediction_points, rows[['centroid_x', 'centroid_y']], atol=1e-3):
            raise ValueError(f'Instance map and CSV centroids disagree for image {index}.')

        truth_matched, prediction_matched = match_centroids(truth_points, prediction_points)
        np.add.at(confusion, (truth_classes[truth_matched] - 1, prediction_classes[prediction_matched]), 1)
        missed = np.delete(truth_classes, truth_matched)
        np.add.at(confusion, (missed - 1, class_count + 1), 1)
        extra = np.delete(prediction_classes, prediction_matched)
        np.add.at(confusion, (class_count, extra), 1)

        iou = pairwise_iou(truth_map, prediction_map, len(truth_classes), prediction_count)
        every_truth = np.ones(len(truth_classes), dtype=bool)
        every_prediction = np.ones(prediction_count, dtype=bool)
        image_counts = panoptic_counts(iou, every_truth, every_prediction)
        mask_counts += image_counts
        # PanNuke protocol: a patch only scores a class it has ground truth for.
        row = {
            'image_index': index,
            'tissue': tissues[index],
            'truth_nuclei': len(truth_classes),
            'predicted_nuclei': prediction_count,
            'centroid_matches': len(truth_matched),
            'binary_pq': panoptic_quality(image_counts)[2] if len(truth_classes) else np.nan,
        }
        for class_id, name in enumerate(CLASS_NAMES, start=1):
            counts = panoptic_counts(iou, truth_classes == class_id, prediction_classes == class_id)
            class_mask_counts[class_id - 1] += counts
            row[f'pq_{name}'] = panoptic_quality(counts)[2] if (truth_classes == class_id).any() else np.nan
        image_rows.append(row)
    return confusion, mask_counts, class_mask_counts, pd.DataFrame(image_rows)


def summarize(confusion, mask_counts, class_mask_counts, images):
    class_count = len(CLASS_NAMES)
    pq_columns = [f'pq_{name}' for name in CLASS_NAMES]
    # PanNuke protocol: average patches within each tissue, then average the 19 tissues.
    tissue = images.groupby('tissue').agg(
        patches=('image_index', 'size'),
        truth_nuclei=('truth_nuclei', 'sum'),
        predicted_nuclei=('predicted_nuclei', 'sum'),
        centroid_matches=('centroid_matches', 'sum'),
        binary_pq=('binary_pq', 'mean'),
        **{column: (column, 'mean') for column in pq_columns},
    )
    tissue['multi_class_pq'] = tissue[pq_columns].mean(axis=1)
    tissue['detection_f1'] = 2 * tissue.centroid_matches / (tissue.truth_nuclei + tissue.predicted_nuclei)
    tissue = tissue.drop(columns=pq_columns + ['centroid_matches'])

    truth_total = confusion[:class_count].sum()
    prediction_total = confusion[:, :class_count + 1].sum()
    matched = confusion[:class_count, :class_count + 1].sum()
    correct = np.trace(confusion[:class_count, 1:class_count + 1])
    mask_detection, mask_segmentation, pooled_pq = panoptic_quality(mask_counts)
    overall = [
        ('Ground-truth nuclei', truth_total),
        ('Predicted nuclei', prediction_total),
        ('Detection precision (centroid)', matched / prediction_total),
        ('Detection recall (centroid)', matched / truth_total),
        ('Detection F1 (centroid)', 2 * matched / (prediction_total + truth_total)),
        ('Detection precision (IoU > 0.5)', mask_counts[0] / prediction_total),
        ('Detection recall (IoU > 0.5)', mask_counts[0] / truth_total),
        ('Detection F1 (IoU > 0.5)', mask_detection),
        ('Mean IoU of mask matches', mask_segmentation),
        ('Class accuracy of centroid matches', correct / matched),
        ('Detection and classification F1 (centroid)', 2 * correct / (prediction_total + truth_total)),
        ('Binary PQ', tissue.binary_pq.mean()),
        ('Multi-class PQ', tissue.multi_class_pq.mean()),
        ('Binary PQ (pooled nuclei)', pooled_pq),
    ]
    overall = pd.DataFrame(overall, columns=['metric', 'value'])

    class_rows = []
    for class_id, name in enumerate(CLASS_NAMES, start=1):
        truth = confusion[class_id - 1].sum()
        predicted = confusion[:, class_id].sum()
        detected = confusion[class_id - 1, :class_count + 1].sum()
        true_positive = confusion[class_id - 1, class_id]
        precision, recall, f1 = scores(true_positive, predicted, truth)
        class_rows.append({
            'class': name,
            'truth_nuclei': truth,
            'predicted_nuclei': predicted,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_recall': detected / truth,
            'class_accuracy_when_detected': true_positive / detected,
            'pq': images.groupby('tissue')[f'pq_{name}'].mean().mean(),
            'pq_pooled': panoptic_quality(class_mask_counts[class_id - 1])[2],
        })
    classes = pd.DataFrame(class_rows)

    confusion = pd.DataFrame(
        confusion,
        index=pd.Index(CLASS_NAMES + ['No ground truth (extra)'], name='ground_truth'),
        columns=[UNTYPED] + CLASS_NAMES + ['Missed'],
    )
    return overall, classes, confusion, tissue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', type=Path, default=BASE / 'data/masks/fold2/masks.npy')
    parser.add_argument('--types', type=Path, default=BASE / 'data/images/fold2/types.npy')
    arguments = parser.parse_args()

    start = time.perf_counter()
    prediction_maps, table = load_predictions()
    masks = np.load(arguments.masks, mmap_mode='r')
    tissues = np.load(arguments.types)
    if masks.shape != (FOLD_SIZE, INPUT_SIZE, INPUT_SIZE, 6) or len(tissues) != FOLD_SIZE:
        raise ValueError('Expected the complete PanNuke Fold 2 masks and tissue types.')
    overall, classes, confusion, tissue = summarize(*evaluate(masks, tissues, prediction_maps, table))
    runtime = time.perf_counter() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT_DIR / 'fold2_overall_metrics.csv', index=False)
    classes.to_csv(OUTPUT_DIR / 'fold2_class_metrics.csv', index=False)
    confusion.to_csv(OUTPUT_DIR / 'fold2_confusion_matrix.csv')
    tissue.to_csv(OUTPUT_DIR / 'fold2_tissue_metrics.csv')
    pd.set_option('display.width', 200)
    print(overall.to_string(index=False))
    print(classes.round(3).to_string(index=False))
    print(confusion.to_string())
    print(tissue.round(3).to_string())
    print(f'Evaluation runtime: {runtime:.1f} seconds')


if __name__ == '__main__':
    main()

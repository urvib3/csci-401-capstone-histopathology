"""Inspect ten PanNuke patches and compare clustering of predicted nuclei.

Class checks use centroid containment, not segmentation-mask matching.
"""
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import radius_neighbors_graph

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

BASE = Path(__file__).resolve().parents[1]
CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Non-neoplastic epithelial']
CLASS_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
PATCH_COUNT = 10
INPUT_SIZE = 256
OUTPUT_SIZE = 164
OFFSET = (INPUT_SIZE - OUTPUT_SIZE) // 2
NEIGHBOR_RADIUS = 40
TARGET_GROUP_SIZE = 3


def load_inputs():
    predictions = pd.read_csv(BASE / 'results/sample_predictions/fold2/hovernet_fold2_predictions.csv')
    predictions = predictions[predictions.image_index.between(0, PATCH_COUNT - 1)].copy()
    images = np.load(BASE / 'data/sample_fold2/images.npy', mmap_mode='r')
    masks = np.load(BASE / 'data/sample_fold2/masks.npy', mmap_mode='r')
    expected_images = {f'image_{i:03d}.png' for i in range(PATCH_COUNT)}
    if set(predictions.image) != expected_images:
        raise ValueError('Expected predictions for patches 000 through 009.')
    if not all(row.image == f'image_{row.image_index:03d}.png' for row in predictions.itertuples()):
        raise ValueError('CSV image indices and filenames disagree.')
    if predictions.duplicated(['image', 'nucleus_id']).any():
        raise ValueError('Nucleus IDs must be unique within each patch.')
    if images.shape != (PATCH_COUNT, INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError('Expected the first ten 256x256 RGB images from Fold 2.')
    if masks.shape != (PATCH_COUNT, INPUT_SIZE, INPUT_SIZE, 6):
        raise ValueError('Expected the matching six-channel ground-truth masks.')
    for row in predictions.itertuples():
        if row.predicted_type_id not in range(1, 6):
            raise ValueError('Expected model class IDs 1 through 5.')
        if row.predicted_type != CLASS_NAMES[row.predicted_type_id - 1]:
            raise ValueError('CSV class names and IDs disagree.')
    return predictions, images, masks


def inspect_classes(patch, coordinates, mask, patch_index):
    candidates = []
    for row, (x, y) in zip(patch.itertuples(), coordinates):
        pixel = mask[int(round(y)), int(round(x)), :5]
        channels = np.flatnonzero(pixel > 0)
        if len(channels) != 1:
            continue
        channel = int(channels[0])
        candidates.append({
            'patch': patch_index,
            'nucleus_id': int(row.nucleus_id),
            'ground_truth_key': (channel, int(pixel[channel])),
            'predicted_type': row.predicted_type,
            'centroid_gt_type': CLASS_NAMES[channel],
            'disagreement': row.predicted_type_id != channel + 1,
        })
    # Exclude multiple predictions associated with the same annotated nucleus.
    counts = Counter(candidate['ground_truth_key'] for candidate in candidates)
    return [candidate for candidate in candidates if counts[candidate['ground_truth_key']] == 1]


def spatial_ward(coordinates):
    graph = radius_neighbors_graph(coordinates, radius=NEIGHBOR_RADIUS, include_self=False)
    _, components = connected_components(graph, directed=False)
    labels = np.empty(len(coordinates), dtype=int)
    next_label = 0
    # Separate components prevent the estimator from adding distant graph edges.
    for component in np.unique(components):
        members = np.flatnonzero(components == component)
        if len(members) == 1:
            local_labels = np.array([0])
        else:
            cluster_count = int(np.ceil(len(members) / TARGET_GROUP_SIZE))
            model = AgglomerativeClustering(
                n_clusters=cluster_count,
                linkage='ward',
                connectivity=graph[members][:, members],
            )
            local_labels = model.fit_predict(coordinates[members])
        labels[members] = local_labels + next_label
        next_label += len(np.unique(local_labels))
    return labels


def cluster_centroids(coordinates):
    cluster_count = int(np.ceil(len(coordinates) / TARGET_GROUP_SIZE))
    kmeans = KMeans(n_clusters=cluster_count, random_state=0, n_init=10)
    dbscan_labels = DBSCAN(eps=NEIGHBOR_RADIUS, min_samples=2).fit_predict(coordinates)
    # Every noise point keeps its own variable, rather than sharing a noise label.
    noise = np.flatnonzero(dbscan_labels == -1)
    first_noise_label = dbscan_labels.max() + 1
    dbscan_labels[noise] = np.arange(first_noise_label, first_noise_label + len(noise))
    return {
        'K-means': kmeans.fit_predict(coordinates),
        'DBSCAN': dbscan_labels,
        'Spatial Ward': spatial_ward(coordinates),
    }


def summarize_groups(method, labels, predicted_types):
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return {
        'method': method,
        'nuclei': len(labels),
        'variables': len(groups),
        'mixed_groups': sum(len(np.unique(predicted_types[group])) > 1 for group in groups),
        'largest_group': max(map(len, groups)),
    }


def plot_inspection(images, masks, patches, errors):
    selected = sorted(set(errors.patch.tolist()) | {9})
    figure, axes = plt.subplots(len(selected), 2, figsize=(8, 4 * len(selected)), squeeze=False)
    for row_index, patch_index in enumerate(selected):
        patch, coordinates, predicted_types = patches[patch_index]
        truth_axis, prediction_axis = axes[row_index]
        for axis in (truth_axis, prediction_axis):
            axis.imshow(np.clip(images[patch_index], 0, 255).astype(np.uint8))
            axis.axis('off')
        truth_axis.set_title(f'Patch {patch_index:03d}: ground truth')
        prediction_axis.set_title(f'Patch {patch_index:03d}: aligned predictions')
        for channel, color in enumerate(CLASS_COLORS):
            class_mask = masks[patch_index, ..., channel]
            for instance in np.unique(class_mask[class_mask > 0]):
                truth_axis.contour(class_mask == instance, levels=[0.5], colors=[color], linewidths=1)
        flagged_ids = set(errors.loc[errors.patch == patch_index, 'nucleus_id'])
        for nucleus, coordinate, class_id in zip(patch.itertuples(), coordinates, predicted_types):
            prediction_axis.scatter(*coordinate, color=CLASS_COLORS[class_id - 1], s=35, edgecolors='black')
            if nucleus.nucleus_id in flagged_ids:
                prediction_axis.annotate(
                    str(nucleus.nucleus_id), coordinate, xytext=(5, 5),
                    textcoords='offset points', fontsize=10, color='white',
                    bbox={'facecolor': 'black', 'alpha': 0.6, 'edgecolor': 'none', 'pad': 1},
                )
        prediction_axis.add_patch(Rectangle(
            (OFFSET, OFFSET), OUTPUT_SIZE, OUTPUT_SIZE,
            fill=False, edgecolor='white', linestyle='--',
        ))
    legend = [Patch(color=color, label=name) for color, name in zip(CLASS_COLORS, CLASS_NAMES)]
    figure.legend(handles=legend, loc='lower center', ncol=3, fontsize=8)
    figure.tight_layout(rect=[0, 0.05, 1, 1])
    figure.savefig(BASE / 'results/sample_predictions/prediction_inspection.png', dpi=120)
    plt.close(figure)


def main():
    predictions, images, masks = load_inputs()
    cluster_rows, class_review, patches = [], [], {}
    for patch_index in range(PATCH_COUNT):
        patch = predictions[predictions.image == f'image_{patch_index:03d}.png']
        coordinates = patch[['centroid_x', 'centroid_y']].to_numpy() + OFFSET
        if not np.isfinite(coordinates).all() or (coordinates < OFFSET).any() or (coordinates >= OFFSET + OUTPUT_SIZE).any():
            raise ValueError('Unexpected crop coordinates.')
        predicted_types = patch.predicted_type_id.to_numpy()
        class_review.extend(inspect_classes(patch, coordinates, masks[patch_index], patch_index))
        for method, labels in cluster_centroids(coordinates).items():
            cluster_rows.append(summarize_groups(method, labels, predicted_types))
        patches[patch_index] = (patch, coordinates, predicted_types)
    summary = pd.DataFrame(cluster_rows).groupby('method', sort=False).agg(
        nuclei=('nuclei', 'sum'), variables=('variables', 'sum'),
        mixed_groups=('mixed_groups', 'sum'), largest_group=('largest_group', 'max'),
    )
    summary['reduction_percent'] = (100 * (1 - summary.variables / summary.nuclei)).round(1)
    summary.to_csv(BASE / 'results/clustering_comparison.csv')
    review = pd.DataFrame(class_review)
    errors = review[review.disagreement]
    plot_inspection(images, masks, patches, errors)
    print(summary.to_string())
    print(f'Unique centroid associations: {len(review)}; class-disagreement flags: {len(errors)}')
    print(errors.drop(columns='ground_truth_key').to_string(index=False))


if __name__ == '__main__':
    main()

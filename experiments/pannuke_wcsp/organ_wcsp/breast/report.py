"""Rebuild eight-family statistics and figures from immutable test confusions."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from families import HERE, SEED, metric, read, write

NAMES = {'pixel': 'Pixel learned table', 'hierarchy': 'Hierarchy V1 / V2',
         'scaling_siblings': 'Scaling and siblings', 'decay': 'Decay schedules',
         'nucleus_pairwise': 'Nucleus pairwise', 'type_counts': 'Type counts',
         'type_set': 'Type set', 'raw_component': 'Raw TP component'}


def scores(matrices, weights):
    counts = (weights @ matrices.reshape(len(matrices), -1)).reshape(-1, 6, 6)
    diagonal = np.diagonal(counts, axis1=1, axis2=2)
    union = counts.sum(1) + counts.sum(2) - diagonal
    return np.divide(diagonal, union, out=np.zeros_like(diagonal, dtype=float), where=union > 0)[:, 1:].mean(1)


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    previous = 0.
    adjusted = {}
    for rank, name in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * pvalues[name]))
        adjusted[name] = previous
    return adjusted


def critical_probability(family, pvalues):
    # Invert the Holm decision for this contrast, holding the other seven
    # zero-null p-values fixed. At zero these intervals give the Holm decision.
    lower = 0.
    upper = 1.
    for iteration in range(60):
        middle = (lower + upper) / 2
        trial = dict(pvalues)
        trial[family] = middle
        if holm(trial)[family] <= .05:
            lower = middle
        else:
            upper = middle
    return lower


def main():
    started = time.perf_counter()
    frozen = read(HERE / 'frozen.json')
    split = read(HERE / 'split.json')
    with np.load(HERE / 'all_confusions.npz') as archive:
        matrices = {name: archive[name] for name in archive.files}
    count = len(split['test'])
    draws = np.random.default_rng(SEED).integers(count, size=(5000, count))
    weights = np.zeros((5000, count), int)
    np.add.at(weights, (np.arange(5000)[:, None], draws), 1)
    bootstrap = {name: scores(matrix, weights) for name, matrix in matrices.items()}
    methods = {name: metric(matrix) for name, matrix in matrices.items()}
    results = {}
    pvalues = {}
    for family in NAMES:
        name = 'breast_' + family
        control = frozen['controls'][family]
        comparisons = {}
        for label, reference in [('matched_control', control), ('raw_tp', 'raw_tp'), ('global', 'global_' + family)]:
            differences = bootstrap[name] - bootstrap[reference]
            patch_difference = np.array([metric(matrix)['bio_miou'] for matrix in matrices[name]])
            patch_difference -= np.array([metric(matrix)['bio_miou'] for matrix in matrices[reference]])
            comparisons[label] = {'method': reference,
                'delta': methods[name]['bio_miou'] - methods[reference]['bio_miou'],
                'ci95': np.quantile(differences, [.025, .975]).tolist(),
                'wins': int((patch_difference > 1e-12).sum()),
                'losses': int((patch_difference < -1e-12).sum()),
                'ties': int((abs(patch_difference) <= 1e-12).sum())}
            if label == 'matched_control':
                tail = min(int((differences <= 0).sum()), int((differences >= 0).sum()))
                pvalues[family] = min(1., 2 * (tail + 1) / 5001)
        results[family] = dict(config=frozen['families'][family], **methods[name], comparisons=comparisons)
    adjusted = holm(pvalues)
    for family, result in results.items():
        critical = critical_probability(family, pvalues)
        # Match the finite-bootstrap +1 tail correction used for p-values.
        tail = max(0., (critical / 2 - 1 / 5001) * 5001 / 5000)
        difference = bootstrap['breast_' + family] - bootstrap[frozen['controls'][family]]
        interval = np.quantile(difference, [tail, 1 - tail]).tolist()
        result['comparisons']['matched_control'].update(p_bootstrap=pvalues[family],
            p_holm=adjusted[family], holm_ci95=interval, holm_raw_threshold=critical)
        result['works'] = bool(interval[0] > 0 and adjusted[family] < .05)
        result['significantly_worse'] = bool(interval[1] < 0 and adjusted[family] < .05)
        result['beats_global'] = bool(result['comparisons']['global']['ci95'][0] > 0)
    write('evaluation.json', {'n_test': count, 'results': results, 'methods': methods,
        'bootstrap': {'replicates': 5000, 'seed': SEED, 'unit': 'paired patches, confusion matrices pooled before IoU',
            'interval': 'Percentile; matched-control intervals invert Holm step-down decisions with other seven zero-null p-values fixed.',
            'pvalue': 'Two-sided percentile bootstrap sign-tail with +1 finite-sample correction. Approximate bootstrap inference.',
            'holm_family_count': 8, 'works': 'Positive Holm test-inverted interval and adjusted p < 0.05; excludes significant harm.'},
        'test_receipt': read(HERE / 'test_receipt.json'),
        'seconds_reporting': time.perf_counter() - started})
    with (HERE / 'per_patch.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_index', 'method', 'biological_miou', 'without_dead'])
        for name, matrix in matrices.items():
            for record, patch in zip(split['test'], matrix):
                values = metric(patch)
                writer.writerow([record['image_index'], name, values['bio_miou'], values['without_dead']])
    draw(results)
    document(results, methods, frozen, split)
    print(json.dumps({family: {key: row[key] for key in ['bio_miou', 'works', 'beats_global']} for family, row in results.items()}, indent=2))


def draw(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 5), layout='constrained')
    for position, (family, row) in enumerate(results.items()):
        comparison = row['comparisons']['matched_control']
        lower, upper = comparison['holm_ci95']
        color = '#18866b' if row['works'] else '#b35e38' if row['significantly_worse'] else '#617b96'
        axis.hlines(position, lower, upper, color=color, linewidth=2)
        axis.plot(comparison['delta'], position, 'o', color=color)
    axis.set_yticks(range(len(results)), [NAMES[family] for family in results])
    axis.invert_yaxis()
    axis.axvline(0, color='#999999', linestyle='--', linewidth=1)
    axis.set(xlabel='Breast winner − matched control: pooled biological mIoU',
             title='Breast test: 169 patches | 5,000 paired bootstrap draws\nHolm test-inverted intervals across eight families')
    figure.savefig(HERE / 'summary.png', dpi=180)
    plt.close(figure)


def document(results, methods, frozen, split):
    lines = ['| Family | Breast mIoU | Control mIoU | Gain | Holm interval | Works | Beats global |',
             '|---|---:|---:|---:|---|---|---|']
    for family, row in results.items():
        comparison = row['comparisons']['matched_control']
        lower, upper = comparison['holm_ci95']
        control = methods[comparison['method']]['bio_miou']
        lines.append(f"| {NAMES[family]} | {row['bio_miou']:.6f} | {control:.6f} | {comparison['delta']:+.6f} | [{lower:+.6f}, {upper:+.6f}] | {'Yes' if row['works'] else 'No'} | {'Yes' if row['beats_global'] else 'No'} |")
    lines += ['', '# Breast-only WCSP: controlled eight-family comparison', '',
        '- **Presentation:** [PanNuke organ overview](../../README.md). **Team review:** [22 November update](../../TEAM_UPDATE_20261122.md).', '',
        '![Gain against matched controls](summary.png)', '',
        f"- **Split:** {len(split['fit'])} fit / {len(split['dev'])} dev / {len(split['test'])} test; seed 20261122; 1,235 previously used IDs excluded.",
        '- Breast only, PanNuke fold 2; full 256×256 patches. Split saved before metrics. Previous experiment directories unchanged.',
        f"- **Raw TP baseline:** {methods['raw_tp']['bio_miou']:.6f}. **Fixed-mask vote:** {methods['vote']['bio_miou']:.6f}.",
        '- Type-set pilot passed the predeclared gate against Raw TP, vote and the global model; remaining seven families then ran. Pilot outcomes preserved in pilot_* files.',
        '- All tables and priors refitted on fit GT only. Nucleus-count bands merged until support ≥10; actual boundaries and counts are in priors_and_tables.json.',
        '- Fixed random 32-patch dev subset for families 1–4 to bound full-resolution sweep time; all 84 dev patches for families 5–8. Exact IDs in expansion_protocol.json.',
        '- One winner per family: highest pooled five-class dev mIoU; within 0.0001 choose the declared simpler/weaker setting. Every row and simplicity tuple saved in development.json.',
        '- Frozen settings written before expansion test. Pilot test results reused, not rerun. Test is protected against accidental reruns.', '',
        '## What was tuned', '', '| Family | Breast dev grid | Matched control |', '|---|---|---|',
        '| Pixel | β = 0.05, 0.10, 0.15, 0.30, 0.50; 24-neighbor learned/biological table | Raw TP |',
        '| Hierarchy | V1 pixel unaries / V2 pixel + block unaries; shared (sibling,parent) = (0.1,0.03), (0.3,0.1), (0.6,0.3); block γ = 0.1,0.3,1 | Raw TP |',
        '| Scaling / siblings | Area/boundary scaling off/on × none/Potts/biological; scaled parent = 0.01,0.03,0.1,0.3,0.5; remaining coefficients from tuned V2 | Breast-tuned V2 |',
        '| Decay | Sibling factors 0.4,0.6,0.8; original parent, sibling, fidelity, adaptive schedules; dev-selected combined schedule | Original constant-weight V1 or V2, using Breast tables, according to block fidelity |',
        '| Nucleus pairwise | Original Potts/hybrid β grid × vote/mean/log_mean × no area/area-weight 0.25/0.75 | Fixed-mask vote |',
        '| Counts | count_model/count_blend × vote/mean/log_mean × strength 0.25,0.5,1,2,4 | Fixed-mask vote |',
        '| Type set | presence/presence_evidence × vote/mean/log_mean × strength 0.25,0.5,1,2,4 | Fixed-mask vote |',
        '| Raw component | Original strength 0,0.1,0.25,0.5,1,2; size limit 128,256,65536; confidence gating off/on | Raw TP |', '',
        'Nucleus area is an individual-type likelihood term, not a third pairwise table. Both Potts and hybrid pairwise models were tested with and without it. Hierarchy blocks are soft dominant-class variables, matching the original implemented objective; no hard majority constraint was added.', '',
        '## Frozen winners and global comparison', '', '| Family | Breast setting | Global mIoU | Breast − global (95% CI) |', '|---|---|---:|---|']
    for family, row in results.items():
        comparison = row['comparisons']['global']
        lower, upper = comparison['ci95']
        configuration = json.dumps(row['config'], separators=(', ', ': '))
        lines.append(f"| {NAMES[family]} | `{configuration}` | {methods['global_' + family]['bio_miou']:.6f} | {comparison['delta']:+.6f} [{lower:+.6f}, {upper:+.6f}] |")
    lines += ['', '- Global configs and tables are pinned to repository commit 19e8df5, not retuned. Hierarchy global V1/V2 follows the selected variant; scaling uses original calibrated scaling; decay uses original final combined schedule.',
        '- Global comparisons change both fit data and selected settings. They do not isolate the causal effect of organ specificity. Absolute scores on this Breast cohort cannot be compared directly with historical mixed-organ scores.', '',
        '## Per-class test IoU', '', '| Method | Neoplastic | Inflammatory | Connective | Dead | Epithelial | mIoU without Dead |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['raw_tp', 'vote'] + ['breast_' + family for family in results]:
        row = methods[name]
        values = row['per_class_iou'] + [row['without_dead']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines += ['', '## Wins / losses / ties', '', '| Family | Against matched control | Against Raw TP | Against global |', '|---|---|---|---|']
    for family, row in results.items():
        cells = []
        for contrast in ['matched_control', 'raw_tp', 'global']:
            comparison = row['comparisons'][contrast]
            cells.append(f"{comparison['wins']} / {comparison['losses']} / {comparison['ties']}")
        lines.append('| ' + NAMES[family] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Exact commands', '', 'From repository root, for a fresh environment:', '', '```sh',
        'git switch organ-wcsp-breast', 'uv venv --python 3.12 .venv',
        'uv pip install -r requirements.txt tiatoolbox numba pyyaml',
        'python3 experiments/pannuke_wcsp/organ_wcsp/breast/fetch_sources.py',
        '# Place fold_2.zip and the verified checkpoint in the paths below.',
        'B=experiments/pannuke_wcsp/organ_wcsp/breast',
        '.venv/bin/python "$B/run_all.py" prepare',
        '.venv/bin/python "$B/run_all.py" infer_dev --gpu',
        '.venv/bin/python "$B/run_all.py" dev',
        '.venv/bin/python "$B/run_all.py" infer_test --gpu',
        '.venv/bin/python "$B/run_all.py" test',
        '# Continue only if the predeclared pilot gate passed:',
        '.venv/bin/python "$B/run_all.py" all_dev',
        '.venv/bin/python "$B/run_all.py" all_test',
        '.venv/bin/python "$B/report.py"', '.venv/bin/python "$B/validate.py"', '```', '',
        '- Existing results are immutable: these commands are a reconstruction recipe for a clean copy, not instructions to overwrite this run.',
        '- Data: benchmarks/pannuke/data/fold_2.zip from https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip. Weights: benchmarks/pannuke/data/cache/models/hovernet_fast-pannuke.pth.',
        '- Required checkpoint SHA256: 789b2be8bdaf532627f584a009f83b3ba06e04c0e55c1635252ff12785ef57e8.',
        '- Actual runtime: uv executable unavailable; an existing Python 3.12 environment had iCloud-evicted packages. Matching local package copies under /tmp/breast-pilot-runtime and isolated Python bypassed stalled cloud reads. See runtime.json. Actual invocation:', '',
        '```sh', 'PYTHONPATH=/tmp/breast-pilot-runtime PYTHONPYCACHEPREFIX=/tmp/capstone-pycache \\',
        '  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -S -u "$B/run_all.py" all_dev',
        '# Same prefix for all_test, report.py and validate.py.', '```', '',
        '## Timings and caveats', '',
        f"- Expansion fit + dev: {frozen['expansion_development_seconds']:.2f} seconds. Expansion test: {read(HERE / 'test_receipt.json')['seconds']:.2f} seconds.",
        f"- Pilot cached evaluation + bootstrap: {read(HERE / 'pilot_evaluation.json')['seconds_evaluation']:.2f} seconds. Per-candidate timings are included for pixel/hierarchy sweeps.",
        '- Inference: same frozen checkpoint, four overlapping tiles, reflect padding 46, original watershed, Apple MPS. Dev/test receipts contain model/source hashes and IDs. No cropping.',
        '- Patient/slide independence unverified; pretrained checkpoint exposure to Fold 2 unknown. Results describe these patches, not clinical generalization.',
        '- Small dev set, especially the 32-patch hierarchy subset. All evaluated settings are retained; test was not used to tune them.',
        '- Dead has no fit, dev or test nuclei. Smoothed tables retain all labels, but no organ-specific Dead evidence was learned. Fixed five-class mIoU uses IoU=0 for zero union; without-Dead mIoU is also reported.',
        '- Local search solvers audit local minima and nonincreasing energy; global optimality is not claimed. Nucleus type-set/count and raw-star objectives use their original exact subproblem solvers.',
        '- 5,000 paired patch bootstrap draws, seed 20261122. Primary matched-control decisions use Holm across eight families; test-inverted intervals hold the other seven zero-null p-values fixed. These are decision-compatible intervals, not ordinary simultaneous coverage intervals. Raw/global intervals are unadjusted exploratory comparisons.',
        '- “Works” requires a positive Holm interval and adjusted p < 0.05. A negative significant change is harm, not success. Effects near zero are inconclusive, not proof of equivalence.',
        '- The user requested expansion conditional on a successful pilot. This adaptive continuation is disclosed; the final Holm family count still includes all eight families.',
        '- Data, inference caches, downloaded source references and model weights are ignored. Only code, compact configs, statistics and presentation artifacts are committed.', '']
    (HERE / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

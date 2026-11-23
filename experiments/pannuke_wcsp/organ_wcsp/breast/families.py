"""Breast adapters for historical objectives; never execute historical entry points."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
ORIGINALS = CACHE / 'originals'
SEED = 20261122


def read(path):
    return json.loads(Path(path).read_text())


def json_array(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(name, value):
    destination = HERE / name
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=json_array) + '\n')
    temporary.replace(destination)


def imported(name, relative):
    local = HERE.parents[1] / relative
    path = ORIGINALS / relative.replace('/', '+')
    if not path.exists():
        path = local
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def original(relative):
    return read(ORIGINALS / relative.replace('/', '+'))


def metric(matrix):
    pooled = matrix.sum(0) if matrix.ndim == 3 else matrix
    diagonal = np.diag(pooled)
    union = pooled.sum(0) + pooled.sum(1) - diagonal
    values = np.divide(diagonal, union, out=np.zeros(6), where=union > 0)
    return {'bio_miou': float(values[1:].mean()), 'per_class_iou': values[1:].tolist(),
            'without_dead': float(values[[1, 2, 3, 5]].mean())}


def select(rows):
    maximum = max(row['bio_miou'] for row in rows)
    close = [row for row in rows if row['bio_miou'] >= maximum - .0001]
    return min(close, key=lambda row: tuple(row['simplicity']))


def load_pixels(records):
    patches = []
    for record in records:
        index = record['image_index']
        with np.load(CACHE / f'pred_{index:04d}.npz') as archive:
            probability = archive['p'].astype(float)
        with np.load(CACHE / f'gt_{index:04d}.npz') as archive:
            truth = archive['types'][archive['instances']]
        patches.append((index, probability, truth))
    return patches


class Hierarchy:
    def __init__(self):
        self.core = imported('breast_hierarchy_core', 'hierarchy_20261109/core.py')
        self.scaled = imported('breast_scaling_core', 'scaling_siblings_20261109/core.py')
        self.graph = self.core.graph(256)
        tree = ast.parse((ORIGINALS / 'family_decay_20261117+run.py').read_text())
        names = {'settings', 'schedule', 'grid', 'evaluate_patch'}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.decay = {'np': np, 'core': self.core, 'SIZES': np.array(self.core.SIZES, float),
                      'POTTS': 1 - np.eye(6), 'GRAPH': self.graph}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'historical_decay_functions', 'exec'), self.decay)

    def inputs(self, probability, truth):
        initial, unary = self.core.initial_unary(probability, 1.)
        offsets = self.graph['offsets']
        sections = [slice(a, b) for a, b in zip(offsets, offsets[1:])]
        confidence = []
        for block in self.core.block_probabilities(probability)[1:]:
            ranks = np.partition(block, -2, axis=-1)[..., -2:].reshape(-1, 2)
            confidence.append(.25 + .75 * (ranks[:, 1] - ranks[:, 0]))
        return (0, probability, truth, initial, unary, sections, confidence)

    def predict(self, configuration, probability, truth, tables):
        if configuration.get('kind') == 'decay':
            self.decay['TABLES'] = {'hybrid': np.asarray(tables), 'potts': np.repeat((1 - np.eye(6))[None], 6, axis=0)}
            return self.decay['evaluate_patch'](configuration['schedule'], self.inputs(probability, truth))[0]
        sibling = configuration.get('siblings', 'biological')
        used = np.asarray(tables)
        if sibling == 'potts':
            used = np.repeat((1 - np.eye(6))[None], 6, axis=0)
        beta = 0 if sibling == 'none' else configuration['sibling']
        result, audit = self.scaled.solve(probability, used, self.graph, beta,
                                         configuration['parent'], configuration['gamma'],
                                         scaled=configuration.get('scaled', False))
        assert audit['verified_local_minimum'], audit
        return result[:truth.size].reshape(truth.shape)


def evaluate_pixels(config, patches, tables, hierarchy, flat=None):
    started = time.perf_counter()
    matrices = []
    for index, probability, truth in patches:
        if config['kind'] == 'flat':
            prediction, audit = flat.solve(probability, np.asarray(tables), config['beta'])
            assert audit['verified_local_minimum'], audit
        else:
            prediction = hierarchy.predict(config, probability, truth, tables)
        matrices.append(hierarchy.core.confusion(truth, prediction))
    matrices = np.stack(matrices)
    return dict(config=config, **metric(matrices), seconds=time.perf_counter() - started), matrices


def development(presence):
    started = time.perf_counter()
    frozen = read(HERE / 'frozen.json')
    assert read(HERE / 'evaluation.json')['pilot_passed']
    assert 'families' not in frozen, 'Family selection is immutable.'
    for name in ['frozen.json', 'evaluation.json', 'development.json', 'confusions.npz', 'per_patch.csv']:
        shutil.copyfile(HERE / name, HERE / ('pilot_' + name))
    split = read(HERE / 'split.json')
    hierarchy = Hierarchy()
    flat = imported('breast_flat_core', 'pannuke_d5/core.py')
    # Full-resolution hierarchy sweeps are much slower than nucleus WCSP.
    # Predeclare a shared random 32-patch development subset, never selected by score.
    indices = np.random.default_rng(SEED).choice(len(split['dev']), 32, replace=False)
    indices = sorted(indices.tolist())
    pixel_records = [split['dev'][index] for index in indices]
    write('expansion_protocol.json', {'dev_pixel_ids': [row['image_index'] for row in pixel_records],
          'dev_other_ids': [row['image_index'] for row in split['dev']],
          'reason': 'Fixed random 32-patch dev subset for full-resolution pixel/hierarchy/decay sweeps; all 84 dev patches for nucleus and raw-component families.',
          'selection': 'Pooled five-class biological mIoU; within 0.0001 choose declared simplicity tuple.',
          'global_reference_commit': '19e8df5', 'test_exposure': 'Pilot was evaluated before this gated expansion. No expansion candidate uses test metrics.'})
    counts = np.zeros((6, 6), int)
    scale_counts = np.zeros((6, 6, 6), int)
    for row in split['fit']:
        instances, types = presence.nucleus.load_gt(row['image_index'])
        truth = types[instances]
        counts += flat.adjacency(truth)
        for level, side in enumerate(hierarchy.core.SIZES):
            scale_counts[level] += hierarchy.core.counts(hierarchy.core.majority(truth, side))
    fitted_flat = flat.fit(counts)
    fitted_scales = [hierarchy.core.fit_table(count) for count in scale_counts]
    tables = read(HERE / 'priors_and_tables.json')
    tables['pixel'] = fitted_flat
    tables['hierarchy'] = {'final': [table.tolist() for table, detail in fitted_scales],
                           'details': [detail for table, detail in fitted_scales]}
    # Historical fit writes only inside the Breast directory after redirecting its context.
    fitted_instances = presence.nucleus.fit(split['fit'])
    tables['nucleus'] = read(HERE / 'tables.json')
    (HERE / 'tables.json').unlink()
    frequency = np.ones(5)
    for row in split['fit']:
        mask, types = presence.nucleus.load_gt(row['image_index'])
        np.add.at(frequency, types[1:] - 1, 1)
    frequency /= frequency.sum()
    tables['type_frequency'] = frequency.tolist()
    write('priors_and_tables.json', tables)
    pixel_data = load_pixels(pixel_records)
    rows_by_family = read(HERE / 'development.json')
    winners = {}
    controls = {}
    global_configs = {}

    def sweep(family, candidates, used_tables):
        rows = []
        for config, simplicity in candidates:
            row, matrices = evaluate_pixels(config, pixel_data, used_tables, hierarchy, flat)
            row['simplicity'] = list(simplicity)
            rows.append(row)
            print(family, config['name'], row['bio_miou'], round(row['seconds'], 2), flush=True)
        rows_by_family[family] = rows
        chosen = select(rows)
        winners[family] = chosen['config']
        write('development.json', rows_by_family)
        return chosen

    flat_candidates = []
    for beta in [.05, .10, .15, .30, .50]:
        flat_candidates.append((dict(name=f'pixel_beta_{beta:g}', kind='flat', beta=beta), (beta,)))
    sweep('pixel', flat_candidates, fitted_flat['final'])
    controls['pixel'] = 'raw_tp'
    original_flat = original('pannuke_d5/frozen_config.json')
    global_configs['pixel'] = dict(name='global_pixel', kind='flat', beta=original_flat['beta'])
    scale_tables = tables['hierarchy']['final']
    hierarchy_candidates = []
    for sibling, parent in [(.1, .03), (.3, .1), (.6, .3)]:
        for gamma in [0., .1, .3, 1.]:
            config = dict(name=f'hierarchy_s{sibling:g}_p{parent:g}_g{gamma:g}', kind='hierarchy',
                          sibling=sibling, parent=parent, gamma=gamma, scaled=False)
            hierarchy_candidates.append((config, (gamma > 0, sibling, parent, gamma)))
    sweep('hierarchy', hierarchy_candidates, scale_tables)
    for version, use_blocks in [('v1', False), ('v2', True)]:
        candidates = [row for row in rows_by_family['hierarchy'] if (row['config']['gamma'] > 0) == use_blocks]
        controls[version] = select(candidates)['config']
    controls['hierarchy'] = 'raw_tp'
    global_configs['hierarchy'] = dict(name='global_hierarchy', kind='hierarchy', sibling=.3, parent=.1,
                                       gamma=.3 if winners['hierarchy']['gamma'] else 0., scaled=False)
    v2 = controls['v2']
    scaling_candidates = []
    for scaled in [False, True]:
        for sibling in ['none', 'potts', 'biological']:
            parents = [.01, .03, .1, .3, .5] if scaled else [v2['parent']]
            for parent in parents:
                config = dict(v2, name=f'scale_{scaled}_{sibling}_{parent:g}', scaled=scaled,
                              siblings=sibling, parent=parent)
                scaling_candidates.append((config, (sibling != 'none', sibling == 'biological', scaled, parent)))
    sweep('scaling_siblings', scaling_candidates, scale_tables)
    controls['scaling_siblings'] = 'v2'
    global_configs['scaling_siblings'] = dict(name='global_scaled_calibrated', kind='hierarchy',
                                              sibling=.3, parent=.01, gamma=.3, scaled=True, siblings='biological')
    decay_candidates = []
    for factor in [.4, .6, .8]:
        schedule = hierarchy.decay['settings'](f'decay_{factor:g}', beta=[.3] + [.3 * factor ** level for level in [4, 3, 2, 1, 0]])
        decay_candidates.append((dict(name=schedule['name'], kind='decay', schedule=schedule), (0, factor)))
    for index, schedule in enumerate(hierarchy.decay['grid']()):
        if schedule['name'] in ['v1', 'v2', 'old_decay', 'potts']:
            continue
        config = dict(name=schedule['name'], kind='decay', schedule=schedule)
        decay_candidates.append((config, (1, index)))
    sweep('decay', decay_candidates, scale_tables)
    # Combine dev-selected parent, sibling and confidence-fidelity schedules only.
    leaders = {}
    for component in ['parent', 'sibling', 'unary']:
        options = [row for row in rows_by_family['decay'] if row['config']['name'].startswith(component + '_')]
        leaders[component] = select(options)['config']['schedule']
    combined = hierarchy.decay['settings']('combined', beta=leaders['sibling']['beta'],
               parent=leaders['parent']['parent'], gamma=leaders['unary']['gamma'], adaptive=leaders['unary']['adaptive'])
    row, matrices = evaluate_pixels(dict(name='combined', kind='decay', schedule=combined), pixel_data, scale_tables, hierarchy)
    row['simplicity'] = [2, 0]
    rows_by_family['decay'].append(row)
    winners['decay'] = select(rows_by_family['decay'])['config']
    controls['decay'] = 'decay_v2' if any(winners['decay']['schedule']['gamma']) else 'decay_v1'
    global_decay = original('family_decay_20261117/frozen.json')['configurations']
    global_configs['decay'] = dict(name='global_combined', kind='decay', schedule=next(config for config in global_decay if config['name'] == 'combined'))
    print('Decay combined', row['bio_miou'], flush=True)

    presence.BANDS = tables['type_set']['boundaries']
    nucleus_data = presence.records(split['dev'], np.array(tables['type_set']['band_probability']), frequency)
    instance_rows = []
    for config in presence.nucleus.configs():
        if config['beta'] == 0:
            continue
        result = presence.nucleus.evaluate(config, nucleus_data, fitted_instances)[0]
        result['simplicity'] = [config['beta'], config['area_weight'], config['table'] != 'potts', ['vote', 'mean', 'log_mean'].index(config['unary'])]
        instance_rows.append(result)
    rows_by_family['nucleus_pairwise'] = instance_rows
    winners['nucleus_pairwise'] = select(instance_rows)['config']
    controls['nucleus_pairwise'] = 'vote'
    global_configs['nucleus_pairwise'] = original('instances_20261114/frozen.json')['wcsp']
    count_rows = []
    for mode in ['count_model', 'count_blend']:
        for unary in ['vote', 'mean', 'log_mean']:
            for strength in [.25, .5, 1., 2., 4.]:
                config = presence.config(f'{unary}_{mode}_{strength:g}', mode, unary, strength)
                result = presence.evaluate(config, nucleus_data)[0]
                result['simplicity'] = [strength, mode != 'count_model', ['vote', 'mean', 'log_mean'].index(unary)]
                count_rows.append(result)
                print('count', config['name'], result['bio_miou'], flush=True)
    rows_by_family['type_counts'] = count_rows
    winners['type_counts'] = select(count_rows)['config']
    controls['type_counts'] = 'vote'
    global_configs['type_counts'] = original('presence_20261114/frozen.json')['count']
    winners['type_set'] = frozen['type_set']
    controls['type_set'] = 'vote'
    global_configs['type_set'] = original('presence_20261114/frozen.json')['presence']
    raw = imported('breast_raw_star', 'presence_20261114/raw_star.py')
    raw.CACHE = CACHE
    raw_data = list(raw.inputs(split['dev']))
    raw_rows = []
    configurations = [dict(strength=0, limit=256, adaptive=False)]
    for strength in [.1, .25, .5, 1., 2.]:
        for limit in [128, 256, 65536]:
            for adaptive in [False, True]:
                configurations.append(dict(strength=strength, limit=limit, adaptive=adaptive))
    for config in configurations:
        result = raw.evaluate(raw_data, config)[0]
        result['simplicity'] = [config['strength'], config['limit'], config['adaptive']]
        raw_rows.append(result)
        print('raw_component', config, result['bio_miou'], flush=True)
    rows_by_family['raw_component'] = raw_rows
    winners['raw_component'] = select(raw_rows)['config']
    controls['raw_component'] = 'raw_tp'
    global_configs['raw_component'] = original('presence_20261114/raw_only_frozen.json')
    write('development.json', rows_by_family)
    frozen['families'] = winners
    frozen['controls'] = controls
    frozen['global_configs'] = global_configs
    frozen['all_families_frozen_before_expansion_test'] = True
    frozen['expansion_development_seconds'] = time.perf_counter() - started
    frozen['historical_objective_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ORIGINALS.glob('*.py')}
    write('frozen.json', frozen)
    print('ALL EIGHT FAMILIES FROZEN', flush=True)


def testing(presence):
    started = time.perf_counter()
    frozen = read(HERE / 'frozen.json')
    assert len(frozen['families']) == 8
    assert not (HERE / 'all_confusions.npz').exists(), 'Expansion test is immutable.'
    split = read(HERE / 'split.json')
    tables = read(HERE / 'priors_and_tables.json')
    hierarchy = Hierarchy()
    flat = imported('breast_flat_core', 'pannuke_d5/core.py')
    pixels = load_pixels(split['test'])
    with np.load(HERE / 'pilot_confusions.npz') as archive:
        matrices = {name: archive[name] for name in archive.files}
    summaries = {}
    for control in ['v1', 'v2', 'decay_v1', 'decay_v2']:
        if control.startswith('decay_'):
            configuration = dict(name=control, kind='hierarchy', sibling=.3, parent=.1,
                                 gamma=.3 if control == 'decay_v2' else 0., scaled=False)
        else:
            configuration = frozen['controls'][control]
        result, matrix = evaluate_pixels(configuration, pixels, tables['hierarchy']['final'], hierarchy)
        matrices[control] = matrix
        print('test control', control, result['bio_miou'], flush=True)
    global_scales = original('hierarchy_20261109/tables.json')['final']
    for family in ['pixel', 'hierarchy', 'scaling_siblings', 'decay']:
        fit_table = tables['pixel']['final'] if family == 'pixel' else tables['hierarchy']['final']
        global_table = original('pannuke_d5/frozen_config.json')['table'] if family == 'pixel' else global_scales
        for label, config, used in [('breast', frozen['families'][family], fit_table), ('global', frozen['global_configs'][family], global_table)]:
            result, matrix = evaluate_pixels(config, pixels, used, hierarchy, flat)
            matrices[f'{label}_{family}'] = matrix
            print('test', label, family, result['bio_miou'], flush=True)
    presence.BANDS = tables['type_set']['boundaries']
    data = presence.records(split['test'], np.array(tables['type_set']['band_probability']), np.array(tables['type_frequency']))
    global_priors = original('presence_20261114/priors.json')
    presence.BANDS = [8, 20, 40]
    global_data = presence.records(split['test'], np.array(global_priors['band_probability']), np.array(global_priors['type_frequency']))
    for family in ['nucleus_pairwise', 'type_counts']:
        for label, configs, dataset in [('breast', frozen['families'], data), ('global', frozen['global_configs'], global_data)]:
            config = configs[family]
            if family == 'nucleus_pairwise':
                used = tables['nucleus'] if label == 'breast' else original('instances_20261114/tables.json')
                array_tables = {key: np.asarray(used[key]) for key in ['potts', 'hybrid', 'area']}
                matrix = presence.nucleus.evaluate(config, dataset, array_tables)[1]
            else:
                matrix = presence.evaluate(config, dataset)[1]
            matrices[f'{label}_{family}'] = matrix
            print('test', label, family, metric(matrix)['bio_miou'], flush=True)
    raw = imported('breast_raw_star', 'presence_20261114/raw_star.py')
    raw.CACHE = CACHE
    raw_data = list(raw.inputs(split['test']))
    for label, configs in [('breast', frozen['families']), ('global', frozen['global_configs'])]:
        matrices[f'{label}_raw_component'] = raw.evaluate(raw_data, configs['raw_component'])[2]
    np.savez_compressed(HERE / 'all_confusions.npz', **matrices)
    write('test_receipt.json', {'test_ids': [row['image_index'] for row in split['test']],
          'frozen_sha256': hashlib.sha256((HERE / 'frozen.json').read_bytes()).hexdigest(),
          'seconds': time.perf_counter() - started, 'pilot_reused_without_rerun': True})
    print('TEST COMPLETE', flush=True)

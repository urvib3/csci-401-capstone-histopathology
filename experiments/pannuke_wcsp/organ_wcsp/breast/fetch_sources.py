"""Materialize pinned historical references without changing original folders."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
COMMIT = '19e8df5'
BASE = 'https://raw.githubusercontent.com/urvib3/csci-401-capstone-histopathology/' + COMMIT + '/experiments/pannuke_wcsp/'
FILES = [
    'pannuke_d5/core.py', 'pannuke_d5/frozen_config.json',
    'hierarchy_20261109/core.py', 'hierarchy_20261109/frozen_config.json',
    'hierarchy_20261109/tables.json', 'hierarchy_20261109/version_2/frozen_block_config.json',
    'scaling_siblings_20261109/core.py', 'scaling_siblings_20261109/scaling_frozen.json',
    'family_decay_20261117/run.py', 'family_decay_20261117/frozen.json',
    'instances_20261114/frozen.json', 'instances_20261114/tables.json',
    'presence_20261114/raw_star.py', 'presence_20261114/frozen.json',
    'presence_20261114/priors.json', 'presence_20261114/raw_only_frozen.json',
]


def main():
    cache = HERE / 'cache' / 'originals'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for relative in FILES:
        destination = cache / relative.replace('/', '+')
        if not destination.exists():
            with urlopen(BASE + relative, timeout=120) as response:
                content = response.read()
            destination.write_bytes(content)
        manifest[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    reference = HERE / 'cache' / 'global_reference'
    reference.mkdir(parents=True, exist_ok=True)
    for name in ['frozen.json', 'priors.json']:
        (reference / name).write_bytes((cache / ('presence_20261114+' + name)).read_bytes())
    output = HERE / 'source_manifest.json'
    content = {'commit': COMMIT, 'sha256': manifest}
    if output.exists():
        assert json.loads(output.read_text()) == content, 'Pinned reference mismatch.'
    else:
        output.write_text(json.dumps(content, indent=2) + '\n')
    print('Verified', len(FILES), 'pinned historical reference files.')


if __name__ == '__main__':
    main()

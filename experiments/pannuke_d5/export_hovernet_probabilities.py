"""Export actual HoVer-Net probability maps before TIAToolbox's type argmax.

Uses an existing local checkpoint and Fold 2 RGB images; no training, downloads,
CSV confidence conversion, or ground truth. Output covers the central 164x164.
"""
import argparse
import ast
from collections import OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def inference_architecture(source_directory):
    """Load unchanged inference definitions without optional WSI dependencies.

    The official forward/constructor/layer code is compiled directly from the
    installed source. ModelABC adds pre/postprocessing interfaces; its nn.Module
    base suffices here because preprocessing is identity and we call forward.
    All parameters and buffers must still strictly match the checkpoint.
    """
    import torch
    source_directory = Path(source_directory)
    namespace = {'__name__':'pannuke_hovernet_inference','torch':torch,'nn':torch.nn,
                 'F':torch.nn.functional,'np':np,'math':math,'warnings':warnings,
                 'OrderedDict':OrderedDict,'ModelABC':torch.nn.Module}
    manifest = {}
    for filename, names in [
        ('utils.py', {'centre_crop','centre_crop_to_shape','UpSample2x'}),
        ('hovernet.py', {'TFSamepaddingLayer','DenseBlock','ResidualBlock','HoVerNet'}),
    ]:
        path = source_directory/filename
        text = path.read_text()
        tree = ast.parse(text)
        definitions = [node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))
                       and node.name in names]
        if {node.name for node in definitions} != names:
            raise ValueError(f'Missing official inference definitions in {path}')
        for node in definitions:
            if isinstance(node,ast.ClassDef) and node.name=='HoVerNet':
                node.body = [method for method in node.body if
                             isinstance(method,ast.FunctionDef) and method.name in
                             {'__init__','forward','_create_decoder_branch'}]
        module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                                 *definitions],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
        manifest[filename] = hashlib.sha256(text.encode()).hexdigest()
    return namespace['HoVerNet'],manifest


def image_patches(archive_path, indices):
    with zipfile.ZipFile(archive_path) as archive:
        matches = [n for n in archive.namelist() if n.endswith('/images.npy') or n == 'images.npy']
        if len(matches) != 1:
            raise ValueError('Expected exactly one images.npy member.')
        with archive.open(matches[0]) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported NPY version {version}')
            if fortran or dtype.hasobject or len(shape) != 4 or shape[1:] != (256, 256, 3):
                raise ValueError(f'Unsupported RGB array {shape}, {dtype}.')
            start = stream.tell()
            size = int(np.prod(shape[1:])) * dtype.itemsize
            for index in sorted(set(indices)):
                if not 0 <= index < shape[0]:
                    raise ValueError(f'Index {index} outside RGB array {shape}')
                stream.seek(start + index*size)
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError('Truncated image array')
                image = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
                    raise ValueError('RGB image must be finite in the 0..255 range.')
                yield index, image.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--indices', nargs='+', type=int)
    choice.add_argument('--selection', type=Path, help='CSV containing image_index.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold-zip', type=Path, default=ROOT/'benchmarks/pannuke/data/fold_2.zip')
    parser.add_argument('--weights', type=Path, default=ROOT/'benchmarks/pannuke/data/cache/models/hovernet_fast-pannuke.pth')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--architecture-source',type=Path,
                        help='Official tiatoolbox/models/architecture directory; load inference-only definitions to avoid WSI dependencies.')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error('A local HoVer-Net fast PanNuke checkpoint is required; nothing is downloaded.')
    if args.selection:
        with args.selection.open() as f:
            indices = [int(r['image_index']) for r in csv.DictReader(f)]
    else:
        indices = args.indices
    if not indices:
        parser.error('Select at least one image.')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be new or empty')
    print('Loading local HoVer-Net architecture and checkpoint...', flush=True)
    import torch
    if args.architecture_source:
        HoVerNet,source_hashes = inference_architecture(args.architecture_source)
    else:
        from tiatoolbox.models.architecture.hovernet import HoVerNet
        source_hashes = None
    torch.set_num_threads(4)
    model = HoVerNet(num_types=6, mode='fast')
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    with args.weights.open('rb') as checkpoint:
        checkpoint_hash = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
    print('Checkpoint loaded strictly; exporting RGB patches...',flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for index, image in image_patches(args.fold_zip, indices):
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(args.device)
        # Forward normalizes RGB internally. Do not use infer_batch, which
        # discards the type probabilities with an argmax in this installed version.
        with torch.inference_mode():
            raw = model(tensor)
            if raw['np'].shape != (1, 2, 164, 164) or raw['tp'].shape != (1, 6, 164, 164):
                raise ValueError('Unexpected HoVer-Net raw head shapes; alignment cannot be assumed.')
            nucleus = torch.softmax(raw['np'], dim=1)[0, 1].cpu().numpy()
            # Renormalize biological channels 1..5, conditional on nucleus membership.
            # The binary NP head supplies background, rather than counting TP background twice.
            types = torch.softmax(raw['tp'][:, 1:], dim=1)[0].cpu().numpy()
            types_all = torch.softmax(raw['tp'], dim=1)[0].cpu().numpy()
        folder = output/f'image_{index:04d}'
        folder.mkdir()
        np.savez_compressed(folder/'hovernet_probabilities.npz',
                            nucleus_probability=nucleus, type_probability=types,
                            type_probability_all=types_all)
        metadata = {'model':'hovernet_fast-pannuke','checkpoint':str(args.weights.resolve()),
                    'checkpoint_sha256':checkpoint_hash,
                    'image_index':index,'original_shape':[256,256],'probability_shape':[164,164],
                    'crop':[46,210,46,210],'type_order':[1,2,3,4,5],
                    'nucleus_probability':'softmax NP logits, foreground channel',
                    'type_probability':'softmax TP logits over biological channels 1..5; conditional on nucleus',
                    'type_probability_all':'original TP softmax over channels 0..5, including background; no NP multiplication',
                    'type_probability_all_order':[0,1,2,3,4,5],
                    'architecture_source':str(args.architecture_source.resolve()) if args.architecture_source else 'tiatoolbox import',
                    'architecture_source_sha256':source_hashes,
                    'checkpoint_loaded_strictly':True,
                    'ground_truth_used':False,'torch_version':str(torch.__version__)}
        (folder/'probability_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(f'Exported actual probabilities for image {index}', flush=True)


if __name__ == '__main__':
    main()

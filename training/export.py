import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .environment import ACTION_VERSION
from .features import OBSERVATION_VERSION
from .model import BrainPolicy

EXPORT_VERSION = 'fly-poker-brain-v2'


def export_model(model: BrainPolicy, directory: Path, *, progress=None, config=None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    def add(name, value, dtype):
        array = value.detach().cpu().numpy().astype(dtype)
        arrays[name] = array
    for name in ('csr_indptr', 'csr_indices', 'input_idx', 'output_idx', 'signs'):
        add(name, getattr(model, name), '<i4')
    add('weights', model.edge_weights(), '<f4')
    add('leak', torch.sigmoid(model.leak_logit), '<f4')
    for name in ('bias', 'sensory.weight', 'sensory.bias', 'motor_norm.weight',
                 'motor_norm.bias', 'policy.weight', 'policy.bias', 'value.weight', 'value.bias'):
        add(name, model.state_dict()[name], '<f4')
    chunks = []
    descriptors = {}
    offset = 0
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError('non-finite export tensor')
        blob = array.tobytes(order='C')
        descriptors[name] = {'offset': offset, 'length': array.size,
            'shape': list(array.shape), 'dtype': 'int32' if array.dtype.kind == 'i' else 'float32'}
        chunks.append(blob)
        offset += len(blob)
    data = b''.join(chunks)
    manifest = {'format': EXPORT_VERSION, 'binary': 'model.bin', 'sha256': hashlib.sha256(data).hexdigest(),
        'bytes': len(data), **model.metadata(), 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'license': 'CC-BY-NC-4.0', 'layer_norm_epsilon': 1e-5,
        'progress': progress or {}, 'training_config': config or {}, 'tensors': descriptors}
    (directory / 'model.bin').write_bytes(data)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def export_checkpoint(graph_path: Path, checkpoint_path: Path, directory: Path):
    from .data import load_graph
    from .checkpoint import load_checkpoint
    model = BrainPolicy(load_graph(graph_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    restored = load_checkpoint(checkpoint_path, model, optimizer, np.random.default_rng())
    return export_model(model, directory, progress=restored['progress'], config=restored['config'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export a validated v2 checkpoint for local browser inference')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.graph, args.checkpoint, args.output), indent=2))

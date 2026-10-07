import os
import tempfile
from pathlib import Path
import torch
from .environment import ACTION_VERSION
from .features import OBSERVATION_VERSION
from .model import BrainPolicy

FIXED_BUFFERS = ('csr_indptr', 'csr_indices', 'root_ids', 'signs', 'input_idx', 'output_idx')


def save_checkpoint(path: Path, model: BrainPolicy, optimizer, rng, progress: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'fly-poker-checkpoint-v2', 'metadata': {**model.metadata(),
        'observation_version': OBSERVATION_VERSION, 'action_version': ACTION_VERSION,
        'torch_version': str(torch.__version__), 'license': 'CC-BY-NC-4.0'},
        'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'progress': progress, 'config': config}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, model: BrainPolicy, optimizer, rng) -> dict:
    device = next(model.parameters()).device
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload['metadata']
    if payload['format'] != 'fly-poker-checkpoint-v2' or metadata['graph_sha256'] != model.graph_sha:
        raise ValueError('checkpoint graph identity mismatch')
    if metadata['model'] != model.metadata()['model'] or metadata['observation_version'] != OBSERVATION_VERSION or metadata['action_version'] != ACTION_VERSION:
        raise ValueError('checkpoint model or observation/action version mismatch')
    current = model.state_dict()
    for key in FIXED_BUFFERS:
        saved, expected = payload['model_state'][key], current[key]
        if saved.layout == torch.sparse_csr:
            same = all(torch.equal(a, b) for a, b in ((saved.crow_indices(), expected.crow_indices()),
                (saved.col_indices(), expected.col_indices()), (saved.values(), expected.values())))
        else:
            same = torch.equal(saved, expected)
        if not same:
            raise ValueError('checkpoint modifies fixed connectome buffers')
    model.load_state_dict(payload['model_state'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    rng.bit_generator.state = payload['numpy_rng']
    torch.set_rng_state(payload['torch_rng'].cpu())
    if torch.cuda.is_available() and payload['cuda_rng']:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
    return {'progress': payload['progress'], 'config': payload['config']}


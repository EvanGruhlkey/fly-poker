import os
import tempfile
from pathlib import Path
import torch
from .environment import ACTION_VERSION
from .features import OBSERVATION_VERSION, FEATURE_COUNT
from .model import BrainPolicy

FIXED_BUFFERS = ('csr_indptr', 'csr_indices', 'root_ids', 'signs', 'input_idx', 'output_idx')


def save_checkpoint(path: Path, model: BrainPolicy, optimizer, rng, progress: dict, config: dict):
    if model.config.observation_features != FEATURE_COUNT:
        raise ValueError('new checkpoints require the current observation schema')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'fly-poker-checkpoint-v4', 'metadata': {**model.metadata(),
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
    if model.config.observation_features != FEATURE_COUNT:
        raise ValueError('resume requires the current observation schema')
    device = next(model.parameters()).device
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload['metadata']
    if payload['format'] != 'fly-poker-checkpoint-v4' or metadata['graph_sha256'] != model.graph_sha:
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



def migrate_v2(path: Path, model: BrainPolicy) -> dict:
    return _migrate(path, model, 'fly-poker-checkpoint-v2', 'cards-public-v1', 118)


def migrate_v3(path: Path, model: BrainPolicy) -> dict:
    return _migrate(path, model, 'fly-poker-checkpoint-v3', 'cards-history-v2', 250)


def _migrate(path: Path, model: BrainPolicy, source_format: str, source_version: str, width: int) -> dict:
    import hashlib
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    metadata = payload['metadata']
    expected = model.metadata()
    old_config = {**expected['model'], 'observation_features': width}
    if payload['format'] != source_format or metadata['graph_sha256'] != model.graph_sha or metadata['model'] != old_config or metadata['observation_version'] != source_version or metadata['action_version'] != ACTION_VERSION:
        raise ValueError('incompatible migration source')
    state = payload['model_state']
    current = model.state_dict()
    if any(not torch.equal(state[key], current[key]) for key in FIXED_BUFFERS):
        raise ValueError('migration modifies fixed connectome buffers')
    if state['sensory.weight'].shape != (len(model.input_idx), width):
        raise ValueError('invalid migration sensory shape')
    widened = torch.zeros_like(current['sensory.weight'])
    widened[:, :width] = state['sensory.weight']
    model.load_state_dict({**state, 'sensory.weight': widened}, strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('non-finite migration source')
    return {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_observation_version': source_version, 'source_progress': payload.get('progress', {}),
        'optimizer_reset': True, 'history_columns': 'zero initialized' if width == 118 else 'copied', 'added_columns': 'zero initialized'}

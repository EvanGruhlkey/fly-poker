import os
import tempfile
from pathlib import Path
import torch
from .environment import ACTION_VERSION
from .features import OBSERVATION_VERSION

FORMAT = 'fly-poker-nfsp-v1'


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_teacher(path, agents, rng, progress, config):
    atomic_save(path, {'format': FORMAT, 'config': config, 'progress': progress,
                      'observation_version': OBSERVATION_VERSION, 'action_version': ACTION_VERSION,
                      'agents': [agent.state() for agent in agents], 'numpy_rng': rng.bit_generator.state,
                      'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


def read_teacher(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT or payload.get('observation_version') != OBSERVATION_VERSION or payload.get('action_version') != ACTION_VERSION:
        raise ValueError('incompatible teacher checkpoint schema')
    return payload


def restore_teacher(path, agents, rng, config):
    payload = read_teacher(path)
    comparable = lambda value: {key: item for key, item in value.items() if key != 'hands'}
    if comparable(payload['config']) != comparable(config):
        raise ValueError('NFSP resume configuration mismatch')
    for agent, state in zip(agents, payload['agents'], strict=True):
        agent.restore(state)
    rng.bit_generator.state = payload['numpy_rng']
    torch.set_rng_state(payload['torch_rng'])
    if torch.cuda.is_available() and payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['progress']

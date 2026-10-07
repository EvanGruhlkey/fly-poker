import numpy as np
import pytest
import torch
from training.data import tiny_graph
from training.model import BrainPolicy
from training.checkpoint import save_checkpoint, load_checkpoint
from training.environment import Holdem
from training.teacher import equity


def test_teacher_only_samples_unknown_cards_from_public_observation():
    obs = Holdem(10).observe()
    assert 0 <= equity(obs, np.random.default_rng(3), samples=8) <= 1
    assert equity(obs, np.random.default_rng(3), samples=8) == equity(obs, np.random.default_rng(3), samples=8)


def test_checkpoint_resumes_tensor_weights_rng_and_validates_identity(tmp_path):
    model = BrainPolicy(tiny_graph())
    optimizer = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(4)
    path = tmp_path / 'checkpoint.pt'
    progress = {'warm_steps_completed': 2, 'updates_completed': 3}
    save_checkpoint(path, model, optimizer, rng, progress, {'seed': 4})
    expected_random = rng.random()
    saved_weight = model.sensory.weight.detach().clone()
    with torch.no_grad():
        model.sensory.weight.add_(1)
    assert load_checkpoint(path, model, optimizer, rng)['progress'] == progress
    assert torch.equal(model.sensory.weight, saved_weight)
    assert rng.random() == expected_random
    payload = torch.load(path, weights_only=True)
    payload['metadata']['observation_version'] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, model, optimizer, rng)

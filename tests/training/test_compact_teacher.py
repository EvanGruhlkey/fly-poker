import numpy as np
import torch
from training.budget import Budget
from training.dataset import build_dataset
from training.environment import Holdem
from training.compact_teacher import CompactTeacher, loadcompact, run


def test_compact_teacher_masks_actions_and_bounds_value():
    model = CompactTeacher()
    mask = torch.tensor([[True, True, False, False, False, False, False]])
    logits, values = model(torch.zeros(1, 267), mask)
    assert torch.isneginf(logits[0, 2:]).all()
    assert -1 <= values.item() <= 1


def test_compact_training_checkpoint_reloads_legal_policy(tmp_path):
    dataset = tmp_path / 'train.npz'
    validation = tmp_path / 'validation.npz'
    build_dataset(dataset, seed=3, count=4, samples=2, budget=Budget(30), teacher='heuristic')
    build_dataset(validation, seed=4, count=4, samples=2, budget=Budget(30), teacher='heuristic', split='validation')
    output = tmp_path / 'result'
    report = run(dataset, output, seconds=10, updates=1, warm_steps=2, batch=64,
        validation_dataset=validation)
    assert report['progress'] == {'warm_steps': 2, 'updates': 1, 'hands': 64}
    assert np.isfinite(report['validation_loss'])
    policy = loadcompact(output / 'latest.pt')
    obs = Holdem(17).observe()
    probabilities = policy(obs)
    assert probabilities.shape == (7,) and np.isclose(probabilities.sum(), 1)
    assert (probabilities[~np.asarray(obs.mask)] == 0).all()
    payload = torch.load(output / 'latest.pt', weights_only=True)
    assert payload['source_sha256'] == report['source_sha256']
    assert payload['numpy_rng'] and payload['optimizer_state']
    run(dataset, output, seconds=10, updates=2, warm_steps=2, batch=64,
        validation_dataset=validation, resume=True)
    run(dataset, tmp_path / 'continuous', seconds=10, updates=2, warm_steps=2,
        batch=64, validation_dataset=validation)
    resumed = torch.load(output / 'latest.pt', weights_only=True)
    continuous = torch.load(tmp_path / 'continuous/latest.pt', weights_only=True)
    assert resumed['numpy_rng'] == continuous['numpy_rng']
    assert torch.equal(resumed['torch_rng'], continuous['torch_rng'])
    assert all(torch.equal(value, continuous['model_state'][key])
        for key, value in resumed['model_state'].items())

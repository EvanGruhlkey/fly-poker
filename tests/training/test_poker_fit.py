import torch
import pytest
from training.data import tiny_graph
from training.model import BrainPolicy
from training.poker_fit import CONNECTORS, dopamine_loss


def test_probe_fit_cannot_publish_without_heldout_validation(tmp_path):
    from training.poker_fit import run
    with pytest.raises(ValueError, match='held-out promotion'):
        run(tmp_path / 'missing.npz', tmp_path / 'missing.pt', tmp_path / 'run',
            steps=1, batch=1, samples=1, wall_seconds=1, seed=1, lr=.001,
            publish=tmp_path / 'production')


def test_a_good_fly_choice_is_reinforced_and_a_bad_one_is_not():
    logits = torch.zeros(1, 2, requires_grad=True)
    prediction = torch.zeros(1, requires_grad=True)
    rewards = torch.tensor([[-1., 1.]])
    good = dopamine_loss(logits, prediction, rewards, torch.tensor([1]))
    logits.grad = None
    good.backward()
    assert logits.grad[0, 1] < 0
    bad_logits = torch.zeros(1, 2, requires_grad=True)
    bad = dopamine_loss(bad_logits, torch.zeros(1), rewards, torch.tensor([0]))
    bad.backward()
    assert bad_logits.grad[0, 0] > 0


def test_the_action_head_reads_fly_output_neurons():
    torch.manual_seed(2)
    model = BrainPolicy(tiny_graph())
    assert model.policy.weight.shape == (7, len(model.output_idx))
    assert set(CONNECTORS).isdisjoint({'syn_gain', 'leak_logit', 'bias'})
    features = torch.randn(1, model.config.observation_features)
    mask = torch.ones(1, 7, dtype=torch.bool)
    live, _ = model(features, mask)
    dead, _ = model(features, mask, ablation='zero')
    assert not torch.allclose(live, dead)

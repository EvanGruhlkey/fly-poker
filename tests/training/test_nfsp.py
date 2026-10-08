import numpy as np
import torch
from training.nfsp import Memory, NFSPAgent, masked_probabilities


def test_reservoir_is_seeded_and_can_retain_early_items():
    first, second = Memory(4, reservoir=True), Memory(4, reservoir=True)
    for memory in (first, second):
        rng = np.random.default_rng(9)
        for index in range(100):
            memory.add(index, rng)
    assert first.items == second.items
    assert len(first.items) == 4
    assert first.seen == 100
    assert min(first.items) < 96


def test_policy_never_assigns_illegal_probability():
    probabilities = masked_probabilities(torch.tensor([[1., 100., 2.]]), torch.tensor([[True, False, True]]))
    assert probabilities[0, 1].item() == 0
    assert probabilities.sum().item() == 1


def test_best_response_exploration_obeys_mask_and_only_it_enters_reservoir():
    torch.manual_seed(7)
    agent = NFSPAgent(3, 3, hidden=8, capacity=16)
    rng = np.random.default_rng(4)
    features, mask = np.zeros(3, np.float32), np.array([False, True, False])
    for _ in range(10):
        assert agent.act(features, mask, False, 1., rng) == 1
    assert agent.supervised.seen == 0
    assert agent.act(features, mask, True, 1., rng) == 1
    assert agent.supervised.seen == 1

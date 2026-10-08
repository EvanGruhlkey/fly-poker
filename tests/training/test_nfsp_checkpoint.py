import numpy as np
import torch
from training.nfsp import NFSPAgent
from training.nfsp_checkpoint import save_teacher, restore_teacher


def test_checkpoint_preserves_reservoir_replay_and_both_rngs(tmp_path):
    agents = [NFSPAgent(3, 2, hidden=8, capacity=4) for _ in range(2)]
    rng = np.random.default_rng(31)
    features, mask = np.zeros(3, np.float32), np.ones(2, bool)
    agents[0].act(features, mask, True, .5, rng)
    agents[0].transition(features, mask, 0, 1, features, mask, True, rng)
    path = tmp_path / 'teacher.pt'
    save_teacher(path, agents, rng, {'hands': 1}, {'hands': 1})
    expected_rng, expected_torch = rng.random(), torch.rand(3)
    fresh = [NFSPAgent(3, 2, hidden=8, capacity=4) for _ in range(2)]
    assert restore_teacher(path, fresh, rng, {'hands': 2}) == {'hands': 1}
    assert rng.random() == expected_rng
    assert torch.equal(torch.rand(3), expected_torch)
    assert fresh[0].supervised.seen == 1
    assert fresh[0].replay.items[0][3] == 1

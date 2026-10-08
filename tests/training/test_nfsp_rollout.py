import numpy as np
import torch
from training.kuhn import Kuhn
from training.nfsp import NFSPAgent
from training.nfsp_rollout import collect_hand


def test_transitions_reach_same_players_next_decision_or_settlement():
    agents = [NFSPAgent(9, 2, hidden=8, capacity=10) for _ in range(2)]
    choices = [iter([0, 1]), iter([1])]
    modes = [[], []]
    for seat, agent in enumerate(agents):
        def act(features, mask, mode, epsilon, rng, seat=seat):
            modes[seat].append(mode)
            return next(choices[seat])
        agent.act = act
    game = Kuhn(4)
    report = collect_hand(game, agents, np.random.default_rng(7), eta=.5, epsilon=.2)
    first, second = agents[0].replay.items
    assert not first[6]
    assert first[3] == 0
    assert torch.equal(first[4], second[0])
    assert first[4][3].item() == 1
    assert second[6]
    assert second[3] == game.reward(0)
    assert agents[1].replay.items[0][3] == game.reward(1)
    assert report['decisions'] == 3
    assert modes[0][0] == modes[0][1]

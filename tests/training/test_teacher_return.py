from dataclasses import replace
import numpy as np
from training.environment import Holdem
from training.teacher import net_targets, opponent_equity


def test_net_targets_account_for_blinds_and_earlier_investment():
    obs = replace(Holdem(3).observe(), starting_stack=400, stack=320)
    probabilities, value = net_targets(obs, [(1., 0., 1.)])
    assert np.isclose(probabilities.sum(), 1)
    assert value < -80 / 800


def test_opponent_response_strength_uses_public_board():
    obs = replace(Holdem(3).observe(), hole=('2c', '3c'),
                  board=('As', 'Ah', 'Ad', 'Ks', 'Kd'), street=3)
    strong = opponent_equity(obs, ('Ac', '4d'), np.random.default_rng(5), samples=24)
    weak = opponent_equity(obs, ('7c', '8d'), np.random.default_rng(5), samples=24)
    assert strong == 1
    assert weak < strong

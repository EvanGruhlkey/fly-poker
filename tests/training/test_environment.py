import numpy as np
import pytest
from training.environment import Holdem
from training.features import encode, FEATURE_COUNT


def test_private_structured_observation():
    env = Holdem(4, dealer=0)
    obs = env.observe()
    assert len(obs.hole) == 2 and not obs.board
    assert not hasattr(obs, 'opponent_hole') and not hasattr(obs, 'deck')
    features = encode(obs)
    assert features.shape == (FEATURE_COUNT,)
    assert features[:52].sum() == 2
    assert features[52:104].sum() == 0
    assert features[104:108].tolist() == [1, 0, 0, 0]
    assert features[108] == 1


def test_blinds_and_canonical_actor():
    env = Holdem(1, dealer=0)
    assert env.actor == 0 and env.stacks == (398, 396)
    assert env.pot == 6
    env.step(1)
    assert env.actor == 1
    env.step(1)
    assert env.actor == 1 and len(env.observe().board) == 3


def test_legal_masks_are_deduplicated_and_rewards_include_blinds_once():
    env = Holdem(2)
    actions = env.actions()
    executable = [action for action in actions if action is not None]
    assert len(executable) == len(set(executable))
    env.step(0)
    assert env.terminal and env.reward(0) == -0.5 and env.reward(1) == 0.5
    assert sum(env.stacks) == 800 and env.pot == 0
    with pytest.raises(ValueError):
        env.step(1)


def test_many_random_hands_keep_exact_chip_conservation():
    rng = np.random.default_rng(42)
    for seed in range(100):
        env = Holdem(seed, dealer=seed % 2)
        for _ in range(200):
            assert sum(env.stacks) + env.pot == 800
            if env.terminal:
                assert env.reward(0) + env.reward(1) == 0
                break
            mask = env.observe().mask
            env.step(int(rng.choice(np.flatnonzero(mask))))
        else:
            raise AssertionError('hand did not terminate')

def test_unequal_buyins_preserve_total_and_net_reward():
    env = Holdem(81, stacks=(37, 763))
    while not env.terminal:
        env.step(1)
        assert sum(env.stacks) + env.pot == 800
    assert env.reward(0) + env.reward(1) == 0

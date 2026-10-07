import numpy as np
import torch
from training.budget import Budget
from training.data import tiny_graph
from training.model import BrainPolicy
from training.rollout import rollout, train_batch
from training.evaluation import summarize


def test_on_policy_rollout_rewards_and_real_parameter_update():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = BrainPolicy(tiny_graph())
    frozen = BrainPolicy(tiny_graph())
    episodes = rollout(model, frozen, np.random.default_rng(3), Budget(30), seeds=[3, 4, 5, 6],
        learner_seats=[0, 1, 0, 1], opponents=['random', 'check_call', 'equity', 'frozen'])
    assert len(episodes) == 4 and all(-100 <= episode.reward_bb <= 100 for episode in episodes)
    for episode in episodes:
        assert all(obs.mask[action] for obs, action in zip(episode.observations, episode.actions))
    before = model.policy.weight.detach().clone()
    stats = train_batch(model, torch.optim.Adam(model.parameters(), lr=.01), episodes, Budget(30))
    assert stats['decisions'] > 0 and np.isfinite(stats['loss'])
    assert not torch.equal(before, model.policy.weight)


def test_confidence_intervals_use_deck_pair_units():
    report = summarize([-1, 0, 1])
    assert report['mean_bb_per_hand'] == 0 and report['hands'] == 6
    assert report['conclusion'] == 'inconclusive' and report['ci95'][0] < 0 < report['ci95'][1]
    assert summarize([1])['ci95'] is None

def test_bootstrap_interval_is_reproducible_and_pair_clustered():
    from training.evaluation import summarize
    first = summarize([-3, -1, 0, 2, 5])
    assert first == summarize([-3, -1, 0, 2, 5])
    assert first['ci95_method'] == 'paired-deck percentile bootstrap; 5000 resamples'
    assert first['ci95'][0] <= first['mean_bb_per_hand'] <= first['ci95'][1]

def test_variable_stack_value_target_fits_tanh_range():
    from training.rollout import normalized_reward
    assert normalized_reward(199) == .995
    assert normalized_reward(-199) == -.995

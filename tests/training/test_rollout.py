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
        assert len(episode.log_probs) == len(episode.values) == len(episode.actions)
        assert np.isfinite(episode.log_probs).all()
    before = model.policy.weight.detach().clone()
    stats = train_batch(model, torch.optim.Adam(model.parameters(), lr=.01), episodes, Budget(30))
    assert stats['decisions'] > 0 and np.isfinite(stats['loss'])
    assert not torch.equal(before, model.policy.weight)
    assert stats['epochs'] >= 2 and stats['updates'] >= 2
    assert np.isfinite(stats['approx_kl']) and 0 <= stats['clip_fraction'] <= 1


def test_rollout_dealer_and_named_snapshot_preserve_behavior_probabilities():
    from training.rollout import tensor_batch
    torch.manual_seed(11)
    model = BrainPolicy(tiny_graph())
    episodes = rollout(model, model, np.random.default_rng(1), Budget(30),
        seeds=[31, 32], learner_seats=[0, 1], dealers=[1, 0],
        opponents=['snapshot', 'snapshot'], opponent_policies={'snapshot': model})
    for episode in episodes:
        assert episode.observations[0].dealer is False
        with torch.no_grad():
            logits, values = model(*tensor_batch(list(episode.observations), 'cpu'))
            expected = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(episode.actions))
        np.testing.assert_allclose(episode.log_probs, expected.numpy(), atol=1e-6)
        np.testing.assert_allclose(episode.values, values.numpy(), atol=1e-6)


def test_ppo_terminal_return_and_clipping_with_fixed_behavior():
    from training.rollout import Episode
    from training.environment import Holdem
    class ConstantPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(7))
            self.value = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, features, mask, ablation='real'):
            return self.logits.expand(len(features), -1).masked_fill(~mask, -torch.inf), self.value.expand(len(features))
    observation = Holdem(7).observe()
    action = next(i for i, legal in enumerate(observation.mask) if legal)
    model = ConstantPolicy()
    episodes = [Episode((observation, observation), (action, action), 100, 7, 0,
        (-10.0, -10.0), (0.0, 0.0))]
    stats = train_batch(model, torch.optim.SGD(model.parameters(), lr=0), episodes, Budget(30))
    assert stats['value_loss'] == .25
    assert stats['clip_fraction'] == 1.0
    assert stats['advantage_mean'] == 0.0


def test_ppo_connectome_drift_is_bounded_and_kl_stops_repeated_updates():
    torch.manual_seed(13)
    model = BrainPolicy(tiny_graph())
    episodes = rollout(model, model, np.random.default_rng(1), Budget(30),
        seeds=[3, 4], learner_seats=[0, 1], opponents=['random', 'random'])
    anchors = {'syn_gain': model.syn_gain.detach().clone()}
    stats = train_batch(model, torch.optim.SGD(model.parameters(), lr=1), episodes,
        Budget(30), epochs=8, anchors=anchors, drift_limit=1e-5, target_kl=1e-8)
    assert torch.max(torch.abs(model.syn_gain - anchors['syn_gain'])) <= 1.1e-5
    assert stats['early_stop'] is True
    assert stats['epochs'] < 8


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


def test_policy_sampling_is_seeded_per_hand_and_evaluation_is_paired():
    from training.evaluation import evaluate
    torch.manual_seed(7)
    model=BrainPolicy(tiny_graph())
    kwargs=dict(seeds=[2**30+3,2**30+3],learner_seats=[0,1],opponents=['random','random'])
    first=rollout(model,model,np.random.default_rng(1),Budget(30),**kwargs)
    second=rollout(model,model,np.random.default_rng(999),Budget(30),**kwargs)
    assert [(item.actions,item.reward_bb) for item in first]==[(item.actions,item.reward_bb) for item in second]
    reports=evaluate(model,model,Budget(30),seed=7,pairs=2)
    assert reports['paired_gain_vs_random']['mean_bb_per_hand']==0
    assert reports['paired_gain_vs_equity']['mean_bb_per_hand']==0
    assert reports['trained_vs_random']['policy_mode']=='sampled-softmax-temperature-1'
    assert reports['trained_vs_random']['action_counts']

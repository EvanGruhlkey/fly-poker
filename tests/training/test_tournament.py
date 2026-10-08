import numpy as np
import pytest

from training.budget import Budget
from training.environment import Holdem
from training.tournament import TournamentConfig, baseline_policy, summarize, tournament


def test_seed_domains_and_profiles():
    validation = TournamentConfig(pairs=3, seed=42)
    test = TournamentConfig(pairs=3, seed=42, domain='test')
    assert set(validation.deal_seeds).isdisjoint(test.deal_seeds)
    assert validation.stack_profiles == ((400, 400), (80, 720), (200, 600))
    with pytest.raises(ValueError):
        TournamentConfig(stack_profiles=((100, 100),))
    with pytest.raises(ValueError):
        TournamentConfig(domain='train')


def test_bootstrap_constant_and_empty():
    assert summarize([2., 2., 2.])['ci95'] == [2., 2.]
    assert summarize([])['mean_bb_per_hand'] is None
    assert summarize([1.])['ci95'] is None


def test_identical_policies_have_zero_paired_gain():
    policy = baseline_policy('random')
    report = tournament(policy, policy, Budget(30), TournamentConfig(pairs=3),
                        opponents={'passive': baseline_policy('passive')})
    for profile in report['profiles'].values():
        gain = profile['paired_gain_vs_passive']
        assert gain['complete']
        assert gain['pair_returns_bb'] == [0., 0., 0.]
        assert gain['ci95'] == [0., 0.]


def test_budget_expiration_is_incomplete():
    report = tournament(baseline_policy('passive'), baseline_policy('passive'),
                        Budget(0), TournamentConfig(pairs=2),
                        opponents={'passive': baseline_policy('passive')})
    assert not report['complete']
    assert report['budget_exhausted']
    assert all(not entry['complete'] for profile in report['profiles'].values()
               for entry in profile.values())


def test_baselines_respect_legal_actions():
    obs = Holdem(21).observe()
    for name in ('random', 'passive', 'aggressive', 'equity'):
        probs = baseline_policy(name)(obs)
        assert probs.shape == (7,)
        assert probs.sum() == pytest.approx(1)
        assert np.all(probs[~np.array(obs.mask)] == 0)

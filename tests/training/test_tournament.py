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


def test_promotion_requires_final_test_full_panel_and_sample_size():
    from training.tournament import promotion
    positive = {'complete': True, 'ci95': [0.1, 2.], 'paired_samples': 1000}
    panel = ('random', 'passive', 'aggressive', 'equity', 'teacher', 'v2')
    profile = {f'{prefix}_vs_{name}': positive.copy()
               for prefix in ('candidate', 'paired_gain') for name in panel}
    report = {'complete': True, 'config': {'domain': 'test', 'pairs': 1000},
              'profiles': {name: profile.copy() for name in ('400_400', '80_720', '200_600')}}
    assert promotion(report, parity=True)['promoted']
    assert not promotion(report, parity=False)['promoted']
    report['config']['domain'] = 'validation'
    assert not promotion(report, parity=True)['promoted']
    report['config']['domain'] = 'test'
    report['profiles']['400_400']['paired_gain_vs_v2'] = {**positive, 'ci95': [-.1, 2.]}
    assert not promotion(report, parity=True)['promoted']
    assert not promotion({'complete': True, 'profiles': {}}, parity=True)['promoted']


def test_report_includes_bb100_and_seats():
    policy = baseline_policy('passive')
    report = tournament(policy, policy, Budget(30),
                        TournamentConfig(pairs=2, stack_profiles=((400, 400),)),
                        opponents={'passive': policy})
    entry = report['profiles']['400_400']['candidate_vs_passive']
    assert entry['bb_per_100'] == pytest.approx(entry['mean_bb_per_hand'] * 100)
    assert entry['ci95_bb_per_100'] == pytest.approx(np.array(entry['ci95']) * 100)
    assert set(entry['seats']) == {'0', '1'}
    assert entry['action_frequencies']['1'] == 1
    assert entry['aggression_frequency'] == 0


def test_training_seed_summary_uses_training_seeds_as_sample_units():
    from training.tournament import compare_training_seeds
    def report(value):
        return {'complete': True, 'config': {'seed': 1}, 'deal_seeds': [10, 11],
                'profiles': {'400_400': {'candidate_vs_equity': {
                    'complete': True, 'mean_bb_per_hand': value}}}}
    result = compare_training_seeds({1: report(2), 2: report(2), 3: report(2)})
    assert result['profiles']['400_400']['candidate_vs_equity']['ci95'] == [2., 2.]
    assert result['training_seeds'] == [1, 2, 3]
    assert result['sample_unit'] == 'training seed'
    assert result['complete']
    with pytest.raises(ValueError):
        different = report(2)
        different['deal_seeds'] = [12, 13]
        compare_training_seeds({1: report(2), 2: different})


def test_seat_swap_pairs_share_deals_and_candidate_stack(monkeypatch):
    import training.tournament as module
    records = []
    original = module.Holdem
    def record(seed, dealer, stacks):
        env = original(seed, dealer, stacks)
        records.append((seed, dealer, stacks))
        return env
    monkeypatch.setattr(module, 'Holdem', record)
    policy = baseline_policy('passive')
    tournament(policy, policy, Budget(30), TournamentConfig(pairs=1, stack_profiles=((80, 720),)),
               opponents={'passive': policy})
    assert records[0][0] == records[1][0] == records[2][0] == records[3][0]
    assert [r[1] for r in records] == [0, 0, 0, 0]
    assert [r[2] for r in records] == [(80, 720), (720, 80), (80, 720), (720, 80)]


def test_match_carries_stacks_and_alternates_dealer(monkeypatch):
    import training.tournament as module
    stacks_seen = []
    original = module.Holdem
    def record(seed, dealer, stacks):
        stacks_seen.append((dealer, stacks))
        return original(seed, dealer, stacks)
    monkeypatch.setattr(module, 'Holdem', record)
    report = tournament(baseline_policy('passive'), baseline_policy('passive'), Budget(30),
                        TournamentConfig(pairs=1, match_hands=2, stack_profiles=((400, 400),)),
                        opponents={'passive': baseline_policy('passive')})
    assert report['sample_unit'] == 'seat-swapped match pair'
    assert stacks_seen[0] == (0, (400, 400))
    assert stacks_seen[1][0] == 1
    assert stacks_seen[1][1] != (400, 400)
    assert sum(stacks_seen[1][1]) == 800


def test_illegal_probabilities_rejected():
    def invalid(obs):
        return np.ones(7) / 7
    with pytest.raises(ValueError, match='legal probabilities'):
        tournament(invalid, invalid, Budget(30), TournamentConfig(pairs=1),
                   opponents={'passive': baseline_policy('passive')})

def test_evaluation_loader_roundtrip_and_fixed_graph_guard(tmp_path):
    import torch
    from training.checkpoint import save_checkpoint
    from training.data import tiny_graph
    from training.evaluate_cli import load_policy
    from training.model import BrainPolicy
    graph = tiny_graph()
    model = BrainPolicy(graph)
    path = tmp_path / 'policy.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), np.random.default_rng(1), {}, {})
    policy, info = load_policy(path, graph, 'cpu')
    from training.tournament import torch_policy
    expected = torch_policy(model)
    obs = Holdem(1).observe()
    assert np.allclose(policy(obs), expected(obs))
    assert len(info['sha256']) == 64
    payload = torch.load(path, weights_only=True)
    payload['model_state']['signs'][0] *= -1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='fixed connectome'):
        load_policy(path, graph, 'cpu')


def test_final_gate_rejects_missing_teacher_and_small_samples():
    from training.tournament import promotion
    positive = {'complete': True, 'ci95': [.1, 1.], 'paired_samples': 1000}
    panel = ('random', 'passive', 'aggressive', 'equity', 'teacher', 'v2')
    profiles = {stack: {f'{prefix}_vs_{name}': positive.copy()
                        for prefix in ('candidate', 'paired_gain') for name in panel}
                for stack in ('400_400', '80_720', '200_600')}
    report = {'complete': True, 'config': {'domain': 'test', 'pairs': 1000}, 'profiles': profiles}
    del profiles['400_400']['candidate_vs_teacher']
    assert not promotion(report, parity=True)['promoted']
    profiles['400_400']['candidate_vs_teacher'] = {**positive, 'paired_samples': 999}
    assert not promotion(report, parity=True)['promoted']

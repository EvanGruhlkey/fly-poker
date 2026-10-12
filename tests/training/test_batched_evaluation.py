import numpy as np
from training.budget import Budget
from training.data import tiny_graph
from training.model import BrainPolicy
from training.batched_evaluation import evaluate_models
from training.tournament import TournamentConfig


def test_identical_candidate_has_zero_paired_gain_and_full_metrics():
    import torch
    torch.set_num_threads(1)
    model = BrainPolicy(tiny_graph())
    config = TournamentConfig(pairs=3, seed=13, stack_profiles=((80, 720),))
    report = evaluate_models(model, model, Budget(30), config, batch_pairs=2, opponents={'passive': None})
    profile = report['profiles']['80_720']
    assert profile['paired_gain_vs_passive']['pair_returns_bb'] == [0., 0., 0.]
    assert report['complete']
    assert profile['candidate_vs_passive']['hands'] == 6
    assert profile['candidate_vs_passive']['seats']['0']['paired_samples'] == 3
    assert sum(profile['candidate_vs_passive']['action_frequencies'].values()) == 1
    assert profile['candidate_vs_passive']['bb_per_100'] == profile['candidate_vs_passive']['mean_bb_per_hand'] * 100
    assert all(seed < 0 for seed in report['deal_seeds'])


def test_budget_abort_reports_no_fabricated_samples():
    model = BrainPolicy(tiny_graph())
    report = evaluate_models(model, model, Budget(0), TournamentConfig(pairs=2),
                             opponents={'passive': None})
    assert not report['complete']
    assert report['budget_exhausted']
    assert report['profiles']['400_400']['candidate_vs_passive']['paired_samples'] == 0

def test_interrupted_starting_batch_discards_candidate_batch(monkeypatch):
    import training.batched_evaluation as module
    from training.budget import BudgetExceeded
    model = BrainPolicy(tiny_graph())
    calls = []
    def collect(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise BudgetExceeded('partial paired batch')
        return [(1., {'street_0_action_1': 1}, True)] * 4
    monkeypatch.setattr(module, 'collect', collect)
    report = evaluate_models(model, model, Budget(30),
                             TournamentConfig(pairs=2, stack_profiles=((400, 400),)),
                             opponents={'passive': None})
    assert report['budget_exhausted']
    for row in report['profiles']['400_400'].values():
        assert row['paired_samples'] == 0
        assert not row['complete']

def test_teacher_probability_rounding_and_legal_mask_are_safe():
    model = BrainPolicy(tiny_graph())
    report = evaluate_models(model, model, Budget(30),
                             TournamentConfig(pairs=1, stack_profiles=((80, 720),)),
                             opponents={'teacher': None}, samples=1)
    assert report['complete']
    assert report['profiles']['80_720']['paired_gain_vs_teacher']['pair_returns_bb'] == [0.]

def test_resume_matches_uninterrupted_pair_returns():
    import torch
    torch.set_num_threads(1)
    model = BrainPolicy(tiny_graph())
    full_config = TournamentConfig(pairs=4, seed=22, stack_profiles=((80, 720),))
    full = evaluate_models(model, model, Budget(30), full_config, batch_pairs=2, opponents={'passive': None})
    snapshots = []
    from training.budget import BudgetExceeded
    def checkpoint(report):
        import copy
        snapshots.append(copy.deepcopy(report))
        raise BudgetExceeded('bounded worker exit')
    partial = evaluate_models(model, model, Budget(30), full_config, batch_pairs=2,
                              opponents={'passive': None}, checkpoint=checkpoint)
    assert not partial['complete']
    resumed = evaluate_models(model, model, Budget(30), full_config, batch_pairs=2,
                              opponents={'passive': None}, prior=partial)
    assert resumed['complete']
    for key, expected in full['profiles']['80_720'].items():
        assert resumed['profiles']['80_720'][key]['pair_returns_bb'] == expected['pair_returns_bb']
    assert resumed['profiles']['80_720']['candidate_vs_passive']['action_counts'] == full['profiles']['80_720']['candidate_vs_passive']['action_counts']


def test_resume_rejects_configuration_and_artifact_mismatch():
    import pytest
    from training.batched_evaluation import validate_resume
    manifest = {'config': {'pairs': 4}, 'artifacts': {'candidate': {'sha256': 'a'}},
                'baseline_samples': 32, 'inference_batch_pairs': 2}
    validate_resume(manifest, manifest)
    for key, value in (('config', {'pairs': 5}), ('artifacts', {'candidate': {'sha256': 'b'}}),
                       ('baseline_samples', 8), ('inference_batch_pairs', 4)):
        with pytest.raises(ValueError, match='resume'):
            validate_resume(manifest, {**manifest, key: value})

def test_checkpoint_api_resumes_atomic_report_and_pins_checkpoint(tmp_path, monkeypatch):
    import torch
    import pytest
    import training.batched_evaluation as module
    from training.checkpoint import save_checkpoint
    graph = tiny_graph()
    monkeypatch.setattr(module, 'load_graph', lambda path: graph)
    model = BrainPolicy(graph)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), np.random.default_rng(2), {}, {})
    output = tmp_path / 'report.json'
    kwargs = dict(budget_seconds=30, pairs=1, seed=4, device='cpu', samples=1,
                  profiles=((80, 720),), batch_pairs=1)
    report = module.evaluate_checkpoints('unused', path, path, output, **kwargs)
    assert report['complete']
    resumed = module.evaluate_checkpoints('unused', path, path, output, resume=True,
                                         **{**kwargs, 'budget_seconds': 0})
    assert resumed['complete']
    for key, row in report['profiles']['80_720'].items():
        assert resumed['profiles']['80_720'][key]['pair_returns_bb'] == row['pair_returns_bb']
    assert not list(tmp_path.glob('*.partial'))
    with torch.no_grad():
        model.policy.bias[0] += .01
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), np.random.default_rng(2), {}, {})
    with pytest.raises(ValueError, match='resume artifacts'):
        module.evaluate_checkpoints('unused', path, path, output, resume=True, **kwargs)

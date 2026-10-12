import numpy as np
import torch

from training.ablation_experiment import train_controls
from training.budget import Budget
from training.data import tiny_graph
from training.dataset import build_dataset


def test_controls_retrain_on_identical_batches_and_save_loadable_states(tmp_path):
    dataset = tmp_path / 'data.npz'
    build_dataset(dataset, seed=21, count=12, samples=2, budget=Budget(30), teacher='heuristic')
    result = train_controls(tiny_graph(), dataset, tmp_path / 'controls', steps=2,
                            seconds_per_mode=30, seed=7)
    assert set(result['modes']) == {'real', 'shuffle', 'zero', 'mlp'}
    assert result['resource_comparison'] == 'same step limit, wall limit and data batches; FLOPs differ'
    fingerprints = set()
    for mode, entry in result['modes'].items():
        assert entry['actual_steps'] == 2
        assert entry['stop_reason'] == 'step cap'
        assert entry['training_wall_limit_seconds'] == 27
        assert np.isfinite(entry['validation_loss'])
        fingerprints.add(entry['batch_indices_sha256'])
        state = torch.load(entry['checkpoint'], weights_only=True)
        assert state['config']['mode'] == mode
        assert state['config']['initialization'] == 'from scratch'
        assert state['model_state']
    assert len(fingerprints) == 1
    assert not set(result['train_hand_seeds']) & set(result['validation_hand_seeds'])


def test_zero_budget_reports_zero_completed_steps(tmp_path):
    dataset = tmp_path / 'data.npz'
    build_dataset(dataset, seed=21, count=8, samples=1, budget=Budget(30), teacher='heuristic')
    result = train_controls(tiny_graph(), dataset, tmp_path / 'controls', steps=2, seconds_per_mode=0)
    assert all(entry['actual_steps'] == 0 and entry['budget_exhausted'] for entry in result['modes'].values())


def test_wall_budget_regime_uses_shared_long_sequence(tmp_path):
    dataset = tmp_path / 'data.npz'
    build_dataset(dataset, seed=21, count=8, samples=1, budget=Budget(30), teacher='heuristic')
    result = train_controls(tiny_graph(), dataset, tmp_path / 'controls', steps=100000, seconds_per_mode=0)
    assert result['config']['experiment_regime'] == 'wall-budget control'
    assert result['config']['training_budget_fraction'] == .9
    assert np.load(tmp_path / 'controls' / 'batch-indices.npy').shape == (100000,16)
    assert all(entry['stop_reason'] == 'training wall budget' for entry in result['modes'].values())

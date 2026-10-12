import numpy as np
import pytest

from training.budget import Budget, BudgetExceeded
from training.dataset import build_dataset, load_dataset
from training.features import FEATURE_COUNT


def test_dataset_reproducible_and_split_disjoint(tmp_path):
    first = build_dataset(tmp_path / 'a.npz', seed=7, count=4, samples=1, budget=Budget(30), teacher='equity')
    second = build_dataset(tmp_path / 'b.npz', seed=7, count=4, samples=1, budget=Budget(30), teacher='equity')
    heldout = build_dataset(tmp_path / 'v.npz', seed=7, count=4, samples=1, budget=Budget(30), teacher='equity', split='validation')
    for key in ('features', 'masks', 'probabilities', 'values'):
        assert np.array_equal(first[key], second[key])
    assert first['features'].shape == (4, FEATURE_COUNT)
    assert np.allclose(first['probabilities'].sum(axis=1), 1)
    assert np.all(first['probabilities'][~first['masks']] == 0)
    assert set(first['hand_seeds']).isdisjoint(heldout['hand_seeds'])
    from training.environment import Holdem
    assert {abs(int(seed)) for seed in first['hand_seeds']}.isdisjoint(abs(int(seed)) for seed in heldout['hand_seeds'])
    assert Holdem(int(first['hand_seeds'][0]))._deck != Holdem(int(heldout['hand_seeds'][0]))._deck
    loaded = load_dataset(tmp_path / 'a.npz', teacher='equity', seed=7, count=4, samples=1, split='train')
    assert loaded['metadata']['checksum'] == first['metadata']['checksum']
    with pytest.raises(ValueError, match='configuration'):
        load_dataset(tmp_path / 'a.npz', seed=8)


def test_no_partial_dataset_on_expired_budget(tmp_path):
    path = tmp_path / 'failed.npz'
    with pytest.raises(BudgetExceeded):
        build_dataset(path, seed=1, count=1, samples=1, budget=Budget(0))
    assert not path.exists()


def test_dataset_checksum_detects_modified_values(tmp_path):
    path = tmp_path / 'data.npz'
    build_dataset(path, seed=1, count=1, samples=1, budget=Budget(30), teacher='equity')
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays['values'] = arrays['values'] + .1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='checksum'):
        load_dataset(path)

def test_equity_dataset_values_subtract_invested_chips(tmp_path, monkeypatch):
    import training.dataset as module
    invested = []
    def targets(obs, probability):
        invested.append(obs.starting_stack - obs.stack)
        probabilities = np.zeros(7, dtype=np.float32)
        probabilities[1] = 1
        return probabilities, .25
    monkeypatch.setattr(module, 'action_targets', targets)
    result = build_dataset(tmp_path / 'net.npz', seed=2, count=4, samples=1,
                           teacher='equity', budget=Budget(30))
    assert np.allclose(result['values'], .25 - np.asarray(invested) / 800)
    assert result['metadata']['teacher_version'] == 'range-equity-net-v1'


def test_rollout_dataset_uses_net_teacher(tmp_path):
    result = build_dataset(tmp_path / 'rollout.npz', seed=3, count=1, samples=1,
                           teacher='rollout', budget=Budget(30))
    assert result['metadata']['teacher_version'] == 'range-response-net-v2'
    assert np.isfinite(result['values']).all()
    assert load_dataset(tmp_path / 'rollout.npz')['metadata'] == result['metadata']


def test_balanced_pressure_quotas_are_legal_and_split_disjoint(tmp_path):
    datasets = [build_dataset(tmp_path / f'{split}.npz', seed=43, count=36, samples=1,
        budget=Budget(10), teacher='heuristic', split=split, sampling='balanced-pressure')
        for split in ('train', 'validation')]
    for data in datasets:
        streets = data['features'][:, 104:108].argmax(1)
        odds = data['features'][:, 252]
        bins = np.where(odds == 0, 0, np.where(odds <= .25, 1, 2))
        assert np.bincount(streets).tolist() == [9] * 4
        assert np.bincount(bins).tolist() == [12] * 3
        assert np.bincount(streets * 3 + bins).tolist() == [3] * 12
        assert not data['probabilities'][~data['masks']].any()
        assert data['metadata']['sampling'] == 'balanced-pressure'
    assert set(datasets[0]['hand_seeds']).isdisjoint(datasets[1]['hand_seeds'])
    with pytest.raises(BudgetExceeded):
        build_dataset(tmp_path / 'expired.npz', seed=43, count=36, samples=1,
            budget=Budget(0), teacher='heuristic', sampling='balanced-pressure')
    assert not (tmp_path / 'expired.npz').exists()

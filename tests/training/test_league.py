import torch
from training.data import tiny_graph
from training.model import BrainPolicy
from training.league import OpponentLeague


def test_snapshot_is_frozen_and_only_refreshes_on_interval(tmp_path):
    model = BrainPolicy(tiny_graph())
    league = OpponentLeague(model, tmp_path, interval=2, capacity=2)
    original = league.policies()['snapshot_0'].policy.weight.detach().clone()
    with torch.no_grad():
        model.policy.weight.add_(1)
    assert league.refresh(model, 1) is False
    assert torch.equal(league.policies()['snapshot_0'].policy.weight, original)
    assert league.refresh(model, 2) is True
    assert torch.equal(league.policies()['snapshot_0'].policy.weight, original)
    assert torch.equal(league.policies()['snapshot_2'].policy.weight, model.policy.weight)
    assert all(not p.requires_grad for snapshot in league.policies().values() for p in snapshot.parameters())
    assert league.refresh(model, 2) is False


def test_resume_selects_historical_snapshots_before_checkpoint_update(tmp_path):
    model = BrainPolicy(tiny_graph())
    league = OpponentLeague(model, tmp_path, interval=2, capacity=2)
    for update in (2, 4, 6):
        with torch.no_grad():
            model.policy.bias.fill_(update)
        league.refresh(model, update)
    resumed = OpponentLeague(model, tmp_path, interval=2, capacity=2, update=4)
    assert list(resumed.policies()) == ['snapshot_2', 'snapshot_4']
    assert resumed.metadata() == {'interval': 2, 'capacity': 2, 'updates': [2, 4]}
    assert resumed.policies()['snapshot_2'].policy.bias.tolist() == [2.0] * 7
    assert resumed.policies()['snapshot_4'].policy.bias.tolist() == [4.0] * 7
    assert list(tmp_path.glob('*.partial')) == []

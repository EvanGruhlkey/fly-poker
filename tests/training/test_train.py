from pathlib import Path
import torch
from training.budget import Budget
from training.data import tiny_graph
from training.train import TrainConfig, train


def test_experiment_controls_and_adapter_stage(tmp_path):
    TrainConfig(warm_steps=1000, updates=1000).validate()
    report = train(tiny_graph(), tmp_path, TrainConfig(seed=7, warm_steps=1, updates=0,
        eval_pairs=0, batch=2, adapter_steps=2), Budget(30))
    assert not report['edge_magnitudes_changed']
    assert not report['neuron_leaks_changed']
    before = torch.load(tmp_path / 'initial.pt', weights_only=True)['model_state']
    after = torch.load(tmp_path / 'latest.pt', weights_only=True)['model_state']
    assert not torch.equal(before['policy.bias'], after['policy.bias'])
    assert report['value_units'] == 'net terminal BB / 200'


def test_bounded_training_checkpoint_and_resume(tmp_path):
    config = TrainConfig(seed=7, warm_steps=1, updates=1, batch=2, eval_pairs=1)
    report = train(tiny_graph(), tmp_path, config, Budget(30))
    assert report['progress']['warm_steps'] == 1
    assert report['progress']['updates'] == 1
    assert report['sensory_weights_changed']
    saved = torch.load(tmp_path / 'latest.pt', weights_only=True)
    assert saved['metadata']['graph_sha256'] == 'synthetic-test-only'
    resumed = train(tiny_graph(), tmp_path, TrainConfig(seed=7, warm_steps=1,
        updates=2, batch=2, eval_pairs=0, resume=True), Budget(30))
    assert resumed['progress']['updates'] == 2


def test_expired_budget_still_saves_initial_checkpoint(tmp_path):
    report = train(tiny_graph(), tmp_path, TrainConfig(warm_steps=1, updates=1,
        eval_pairs=0), Budget(0))
    assert report['budget_exhausted']
    assert report['progress']['updates'] == 0
    assert (tmp_path / 'latest.pt').exists()


def test_cached_training_keeps_local_checkpoints_without_syncing_every_eight_steps(tmp_path):
    from training.dataset import build_dataset
    dataset = tmp_path / 'labels.npz'
    build_dataset(dataset, seed=8, count=16, samples=1, budget=Budget(30), teacher='heuristic')
    commits = []
    output = tmp_path / 'run'
    report = train(tiny_graph(), output, TrainConfig(seed=8, warm_steps=32, updates=0,
        eval_pairs=0, batch=2, adapter_steps=16, dataset=str(dataset)), Budget(30), commit=lambda: commits.append(1))
    assert report['progress']['warm_steps'] == 32
    assert torch.load(output / 'latest.pt', weights_only=True)['progress']['warm_steps'] == 32
    assert len(commits) <= 3
    adapter = torch.load(output / 'adapter.pt', weights_only=True)
    imitation = torch.load(output / 'imitation.pt', weights_only=True)
    assert adapter['progress']['warm_steps'] == 16
    assert imitation['progress']['warm_steps'] == 32 and imitation['progress']['updates'] == 0
    initial = torch.load(output / 'initial.pt', weights_only=True)['model_state']
    assert torch.equal(adapter['model_state']['syn_gain'], initial['syn_gain'])


def test_interrupted_ppo_rolls_back_parameters_optimizer_and_rng(tmp_path, monkeypatch):
    import importlib
    from training.budget import BudgetExceeded
    module = importlib.import_module('training.train')
    original = module.train_batch
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise BudgetExceeded('interrupted after an optimizer update')
    common = dict(seed=19, warm_steps=0, updates=1, batch=2, eval_pairs=0)
    monkeypatch.setattr(module, 'train_batch', interrupted)
    report = train(tiny_graph(), tmp_path / 'resumed', TrainConfig(**common), Budget(30))
    assert report['progress']['updates'] == 0
    monkeypatch.setattr(module, 'train_batch', original)
    train(tiny_graph(), tmp_path / 'resumed', TrainConfig(**common, resume=True), Budget(30))
    train(tiny_graph(), tmp_path / 'continuous', TrainConfig(**common), Budget(30))
    first = torch.load(tmp_path / 'resumed/latest.pt', weights_only=True)
    second = torch.load(tmp_path / 'continuous/latest.pt', weights_only=True)
    assert torch.equal(first['torch_rng'], second['torch_rng'])
    assert first['numpy_rng'] == second['numpy_rng']
    assert all(torch.equal(first['model_state'][key], value) for key, value in second['model_state'].items())

def test_resume_matches_uninterrupted_update_trajectory(tmp_path):
    graph = tiny_graph()
    common = dict(seed=19, warm_steps=1, batch=2, eval_pairs=0)
    train(graph, tmp_path / 'continuous', TrainConfig(**common, updates=2), Budget(30))
    train(graph, tmp_path / 'resumed', TrainConfig(**common, updates=1), Budget(30))
    train(graph, tmp_path / 'resumed', TrainConfig(**common, updates=2, resume=True), Budget(30))
    original = torch.load(tmp_path / 'continuous/latest.pt', weights_only=True)['model_state']
    resumed = torch.load(tmp_path / 'resumed/latest.pt', weights_only=True)['model_state']
    for key in original:
        if original[key].layout == torch.strided:
            assert torch.equal(original[key], resumed[key])


def test_rollout_labels_require_measured_teacher_improvement(tmp_path):
    import pytest
    from training.dataset import build_dataset
    path = tmp_path / 'rollout.npz'
    build_dataset(path, seed=3, count=4, samples=2, budget=Budget(30))
    with pytest.raises(ValueError, match='passing benchmark'):
        train(tiny_graph(), tmp_path / 'run', TrainConfig(warm_steps=1, updates=0,
            eval_pairs=0, dataset=str(path)), Budget(30))
    assert not (tmp_path / 'run').exists()


def test_resume_rejects_replaced_dataset_at_same_path(tmp_path):
    import pytest
    from training.dataset import build_dataset
    path = tmp_path / 'data.npz'
    build_dataset(path, seed=3, count=4, samples=2, teacher='equity', budget=Budget(30))
    config = dict(seed=3, warm_steps=1, updates=0, eval_pairs=0, dataset=str(path))
    train(tiny_graph(), tmp_path / 'run', TrainConfig(**config), Budget(30))
    build_dataset(path, seed=4, count=4, samples=2, teacher='equity', budget=Budget(30))
    with pytest.raises(ValueError, match='dataset checksum'):
        train(tiny_graph(), tmp_path / 'run', TrainConfig(**config, resume=True), Budget(30))



def test_migrated_training_keeps_parent_and_records_new_progress(tmp_path):
    from training.model import BrainPolicy, ModelConfig
    old=BrainPolicy(tiny_graph(),ModelConfig(observation_features=118))
    source=tmp_path/'v2.pt'
    torch.save({'format':'fly-poker-checkpoint-v2','metadata':{**old.metadata(),
        'observation_version':'cards-public-v1','action_version':'holdem-seven-v1'},
        'model_state':old.state_dict(),'progress':{'hands':2048}},source)
    data=source.read_bytes()
    config=TrainConfig(seed=18,warm_steps=1,updates=1,batch=2,eval_pairs=1,init_from_v2=str(source))
    report=train(tiny_graph(),tmp_path/'v3',config,Budget(30))
    assert source.read_bytes()==data
    assert report['progress']['hands']==2
    assert report['migration']['source_progress']['hands']==2048
    assert report['starting_policy']=='migrated-v2'
    assert report['evaluation']['paired_gain_vs_random']['complete']

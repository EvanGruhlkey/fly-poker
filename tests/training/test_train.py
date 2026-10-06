from pathlib import Path
import torch
from training.budget import Budget
from training.data import tiny_graph
from training.train import TrainConfig, train


def test_bounded_training_checkpoint_and_resume(tmp_path):
    config = TrainConfig(seed=17, warm_steps=1, updates=1, batch=2, eval_pairs=1)
    report = train(tiny_graph(), tmp_path, config, Budget(30))
    assert report['progress']['warm_steps'] == 1
    assert report['progress']['updates'] == 1
    assert report['sensory_weights_changed']
    saved = torch.load(tmp_path / 'latest.pt', weights_only=True)
    assert saved['metadata']['graph_sha256'] == 'synthetic-test-only'
    resumed = train(tiny_graph(), tmp_path, TrainConfig(seed=17, warm_steps=1,
        updates=2, batch=2, eval_pairs=0, resume=True), Budget(30))
    assert resumed['progress']['updates'] == 2


def test_expired_budget_still_saves_initial_checkpoint(tmp_path):
    report = train(tiny_graph(), tmp_path, TrainConfig(warm_steps=1, updates=1,
        eval_pairs=0), Budget(0))
    assert report['budget_exhausted']
    assert report['progress']['updates'] == 0
    assert (tmp_path / 'latest.pt').exists()

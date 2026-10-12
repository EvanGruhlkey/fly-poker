import json
import numpy as np
import pytest
import torch

from training.ablation_experiment import build_control, MODES
from training.control_evaluation import ControlPolicy, evaluate_controls, load_control
from training.data import tiny_graph
from training.environment import Holdem
from training.features import OBSERVATION_VERSION
from training.rollout import infer


@pytest.mark.parametrize('mode', MODES)
def test_inference_preserves_trained_ablation(mode, monkeypatch):
    model = build_control(tiny_graph(), mode)
    calls = []
    original = model.forward
    def forward(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', forward)
    actions = infer(ControlPolicy(model, mode), [Holdem(3).observe()], greedy=True)
    assert len(actions) == 1
    assert calls == ([{}] if mode == 'mlp' else [{'ablation': mode}])


def write_controls(directory):
    directory.mkdir()
    for mode in MODES:
        model = build_control(tiny_graph(), mode)
        torch.save({'format':'fly-poker-architecture-control-v1',
            'config': {'mode':mode,'graph_sha256':tiny_graph().sha256,
                'observation_version':OBSERVATION_VERSION,'dataset_checksum':'same-cache',
                'seed':31,'steps':64,'batch_indices_sha256':'same-batches'},
            'model_state':model.state_dict()}, directory / f'{mode}.pt')


def test_evaluation_saves_all_incomplete_reports_with_provenance(tmp_path, monkeypatch):
    import training.control_evaluation as module
    directory = tmp_path / 'controls'
    write_controls(directory)
    monkeypatch.setattr(module, 'load_graph', lambda path: tiny_graph())
    monkeypatch.setattr(module, 'load_model', lambda *args: (build_control(tiny_graph(),'real'), {'sha256':'v2-source'}))
    output = tmp_path / 'evaluation'
    result = evaluate_controls('graph.npz',directory,'v2.pt',output,seconds_per_mode=0,pairs=1,device='cpu')
    assert not result['complete']
    for mode in MODES:
        report = json.loads((output / f'{mode}.json').read_text())
        assert not report['complete']
        assert report['artifacts']['control']['sha256']
        assert report['artifacts']['control']['config']['dataset_checksum'] == 'same-cache'
        assert set(report['profiles']) == {'400_400','80_720','200_600'}
        assert 'candidate_vs_v2' in report['profiles']['400_400']


def test_loading_rejects_wrong_graph(tmp_path):
    directory = tmp_path / 'controls'
    write_controls(directory)
    path = directory / 'real.pt'
    payload = torch.load(path, weights_only=True)
    payload['config']['graph_sha256'] = 'wrong'
    torch.save(payload,path)
    with pytest.raises(ValueError):
        load_control(path,tiny_graph(),'cpu')

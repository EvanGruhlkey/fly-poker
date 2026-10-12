import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_app(monkeypatch):
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()
        def pip_install(self, *args):
            return self
        def add_local_dir(self, *args, **kwargs):
            return self
    class App:
        def __init__(self, name):
            pass
        def function(self, **resources):
            def decorate(function):
                function.resources = resources
                function.remote = function
                return function
            return decorate
        def local_entrypoint(self):
            return lambda function: function
    monkeypatch.setitem(__import__('sys').modules, 'modal', SimpleNamespace(App=App,
        Image=Image, Volume=SimpleNamespace(from_name=lambda *args, **kwargs: object())))
    spec = importlib.util.spec_from_file_location('training_modal_test', Path(__file__).parents[2] / 'modalapp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l4_workers_and_explicit_cpu_dispatch(monkeypatch):
    module = load_app(monkeypatch)
    calls = []
    for name in ('smoke', 'pilot', 'smoke_gpu', 'pilot_gpu', 'poker', 'poker_gpu'):
        worker = getattr(module, name)
        monkeypatch.setattr(worker, 'remote', lambda *args, selected=name: calls.append(selected) or {})
        assert worker.resources['max_containers'] == 1
        assert worker.resources['timeout'] == 1200
        assert worker.resources['retries'] == 0
        assert worker.resources.get('gpu') == ('L4' if name.endswith('_gpu') else None)
    module.main(mode='smoke', accelerator='gpu')
    module.main(mode='train', accelerator='gpu')
    module.main(mode='train', accelerator='cpu')
    module.main(mode='poker', accelerator='gpu')
    module.main(mode='poker', accelerator='cpu')
    assert calls == ['smoke_gpu', 'pilot_gpu', 'pilot', 'poker_gpu', 'poker']


def test_control_dispatch_resources_and_caps(monkeypatch):
    import pytest
    module = load_app(monkeypatch)
    calls = []
    worker = module.controls_gpu
    assert {key: worker.resources[key] for key in ('gpu','cpu','memory','max_containers','timeout','retries')} == {
        'gpu':'L4','cpu':2,'memory':8192,'max_containers':1,'timeout':900,'retries':0}
    monkeypatch.setattr(worker, 'remote', lambda *args: calls.append(args) or {})
    module.main(mode='controls')
    assert calls == [(31,64,90)]
    module.main(mode='controls', seed=42, steps=100000, wall_seconds=120)
    assert calls[-1] == (42,100000,120)
    for controls in ({'steps':100001}, {'wall_seconds':121}, {'steps':-1}, {'wall_seconds':-1}):
        with pytest.raises(ValueError):
            module.main(mode='controls', **controls)
    assert len(calls) == 2


def test_control_evaluation_dispatch_and_resources(monkeypatch):
    import pytest
    module = load_app(monkeypatch)
    calls = []
    worker = module.control_evaluate_gpu
    assert {key:worker.resources[key] for key in ('gpu','cpu','memory','timeout','max_containers','retries')} == {
        'gpu':'L4','cpu':4,'memory':8192,'timeout':1200,'max_containers':1,'retries':0}
    monkeypatch.setattr(worker,'remote',lambda *args: calls.append(args) or {})
    module.main(mode='control-evaluate',eval_pairs=32)
    assert calls == [(31,32,120)]
    for controls in ({'wall_seconds':301},{'eval_pairs':10001},{'eval_pairs':0}):
        with pytest.raises(ValueError):
            module.main(mode='control-evaluate',**controls)
    assert len(calls) == 1

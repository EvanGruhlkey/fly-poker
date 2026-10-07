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
    for name in ('smoke', 'pilot', 'smoke_gpu', 'pilot_gpu'):
        worker = getattr(module, name)
        monkeypatch.setattr(worker, 'remote', lambda *args, selected=name: calls.append(selected) or {})
        assert worker.resources['max_containers'] == 1
        assert worker.resources['timeout'] == 1200
        assert worker.resources['retries'] == 0
        assert worker.resources.get('gpu') == ('L4' if name.endswith('_gpu') else None)
    module.main(mode='smoke', accelerator='gpu')
    module.main(mode='train', accelerator='gpu')
    module.main(mode='train', accelerator='cpu')
    assert calls == ['smoke_gpu', 'pilot_gpu', 'pilot']

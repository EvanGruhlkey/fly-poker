from pathlib import Path
import modal

app = modal.App('fly-poker-training')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'numpy==2.3.3', 'pokerkit==0.7.6')
    .add_local_dir(str(Path(__file__).parent / 'training'), remote_path='/root/training', copy=True))
volume = modal.Volume.from_name('fly-poker-artifacts', create_if_missing=True)


def run_smoke(device: str, seed: int, batch: int, iterations: int):
    import torch
    from training.data import fetch
    from training.benchmark import benchmark
    torch.manual_seed(seed)
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    return benchmark(str(path), batch=batch, iterations=iterations, device=device)


def run_pilot(device: str, seed: int, warm_steps: int, updates: int, batch: int,
              eval_pairs: int, wall_seconds: int, resume: bool, init_from_v2: str | None = None,
              experiment: str = ''):
    from training.budget import Budget
    from training.data import fetch, load_graph
    from training.train import TrainConfig, train
    if not 1 <= wall_seconds <= 900:
        raise ValueError('wall_seconds must be between 1 and 900')
    budget = Budget(wall_seconds)
    import json
    controls = json.loads(experiment) if experiment else {}
    run_name = controls.pop('run_name', f'brain-v4-seed-{seed}')
    if not run_name or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for char in run_name):
        raise ValueError('invalid experiment run name')
    config = TrainConfig(**{**dict(seed=seed, warm_steps=warm_steps, updates=updates,
        batch=batch, eval_pairs=eval_pairs, resume=resume, init_from_v2=init_from_v2), **controls})
    config.validate()
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    try:
        return train(load_graph(path), Path(f'/artifacts/runs/{run_name}'), config,
            budget, device=device, commit=volume.commit)
    finally:
        volume.commit()


@app.function(image=image, cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def smoke(seed: int = 17, batch: int = 8, iterations: int = 3):
    return run_smoke('cpu', seed, batch, iterations)


@app.function(image=image, gpu='L4', cpu=2, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def smoke_gpu(seed: int = 17, batch: int = 8, iterations: int = 3):
    return run_smoke('cuda', seed, batch, iterations)


@app.function(image=image, cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def pilot(seed: int = 17, warm_steps: int = 64, updates: int = 32, batch: int = 8,
          eval_pairs: int = 15, wall_seconds: int = 900, resume: bool = False, init_from_v2: str | None = None, experiment: str = ''):
    return run_pilot('cpu', seed, warm_steps, updates, batch, eval_pairs, wall_seconds, resume, init_from_v2, experiment)


@app.function(image=image, gpu='L4', cpu=2, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def pilot_gpu(seed: int = 17, warm_steps: int = 64, updates: int = 32, batch: int = 8,
              eval_pairs: int = 15, wall_seconds: int = 900, resume: bool = False, init_from_v2: str | None = None, experiment: str = ''):
    return run_pilot('cuda', seed, warm_steps, updates, batch, eval_pairs, wall_seconds, resume, init_from_v2, experiment)


def run_poker(device: str, seed: int, steps: int, batch: int, samples: int, wall_seconds: int, lr: float, entropy: float):
    from training.data import fetch
    from training.poker_fit import run
    if not 1 <= wall_seconds <= 900:
        raise ValueError('wall_seconds must be between 1 and 900')
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    try:
        return run(path, Path('/artifacts/runs/poker-gpu-anneal/latest.pt'), Path('/artifacts/runs/poker-gpu-anneal-2'),
            steps=steps, batch=batch, samples=samples, wall_seconds=wall_seconds, seed=seed, lr=lr,
            publish=None, device=device, entropy=entropy)
    finally:
        volume.commit()


@app.function(image=image, gpu='L4', cpu=2, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def poker_gpu(seed: int = 17, steps: int = 2000, batch: int = 8, samples: int = 8,
              wall_seconds: int = 600, lr: float = 0.003, entropy: float = 0.1):
    return run_poker('cuda', seed, steps, batch, samples, wall_seconds, lr, entropy)


@app.function(image=image, cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def poker(seed: int = 17, steps: int = 32, batch: int = 4, samples: int = 8,
          wall_seconds: int = 120, lr: float = 0.003, entropy: float = 0.1):
    return run_poker('cpu', seed, steps, batch, samples, wall_seconds, lr, entropy)


@app.function(image=image, gpu='L4', cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def evaluate_gpu(candidate: str, output: str, pairs: int = 128, seed: int = 731,
                 wall_seconds: int = 900, domain: str = 'validation', samples: int = 16, resume: bool = False):
    from training.batched_evaluation import evaluate_checkpoints
    from training.data import fetch
    if not 1 <= wall_seconds <= 900 or not 1 <= pairs <= 10000:
        raise ValueError('evaluation exceeds bounded experiment controls')
    graph = fetch(Path('/artifacts/data/full.npz'))
    try:
        report = evaluate_checkpoints(graph, Path(candidate),
            Path('/artifacts/runs/brain-v2-seed-17/latest.pt'), Path(output),
            budget_seconds=wall_seconds, pairs=pairs, seed=seed, samples=samples,
            device='cuda', domain=domain, batch_pairs=8, resume=resume, commit=volume.commit)
        return {'complete': report['complete'], 'elapsed_seconds': report['elapsed_seconds'],
                'promotion': report['promotion'], 'output': output}
    finally:
        volume.commit()


@app.function(image=image, gpu='L4', cpu=2, memory=8192, max_containers=1, timeout=900,
              retries=0, volumes={'/artifacts': volume})
def controls_gpu(seed: int = 31, steps: int = 64, wall_seconds: int = 90):
    from training.ablation_experiment import train_controls
    from training.data import fetch, load_graph
    if not 1 <= steps <= 100000 or not 1 <= wall_seconds <= 120:
        raise ValueError('controls require 1 to 100000 steps and 1 to 120 seconds per mode')
    try:
        graph = fetch(Path('/artifacts/data/full.npz'))
        return train_controls(load_graph(graph), Path('/artifacts/datasets/heuristic-train.npz'),
            Path(f'/artifacts/runs/architecture-controls-seed-{seed}'), steps=steps,
            seconds_per_mode=wall_seconds, seed=seed, device='cuda',
            validation_dataset=Path('/artifacts/datasets/heuristic-validation.npz'))
    finally:
        volume.commit()


@app.function(image=image, gpu='L4', cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def control_evaluate_gpu(seed: int = 31, pairs: int = 32, wall_seconds: int = 120):
    from training.control_evaluation import evaluate_controls
    from training.data import fetch
    if not 1 <= wall_seconds <= 300 or not 1 <= pairs <= 10000:
        raise ValueError('control evaluation exceeds bounded experiment limits')
    try:
        graph = fetch(Path('/artifacts/data/full.npz'))
        directory = Path(f'/artifacts/runs/architecture-controls-seed-{seed}')
        return evaluate_controls(graph, directory,
            Path('/artifacts/runs/brain-v2-seed-17/latest.pt'), directory / 'validation',
            seconds_per_mode=wall_seconds, pairs=pairs, device='cuda', commit=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(mode: str = 'smoke', seed: int = -1, batch: int = 8, iterations: int = 3,
         warm_steps: int = 64, updates: int = 32, eval_pairs: int = 15,
         wall_seconds: int = 0, resume: bool = False, accelerator: str = 'cpu', init_from_v2: str = '',
         steps: int = 0, samples: int = 8, lr: float = 0.003, entropy: float = 0.1, experiment_config: str = '',
         candidate: str = '', output: str = '', domain: str = 'validation'):
    import json
    seed = (31 if mode in ('controls', 'control-evaluate') else 17) if seed == -1 else seed
    steps = (64 if mode == 'controls' else 2000) if steps == 0 else steps
    wall_seconds = (90 if mode == 'controls' else 120 if mode == 'control-evaluate' else 900) if wall_seconds == 0 else wall_seconds
    if accelerator not in ('cpu', 'gpu'):
        raise ValueError('accelerator must be cpu or gpu')
    if mode == 'smoke':
        worker = smoke_gpu if accelerator == 'gpu' else smoke
        result = worker.remote(seed, batch, iterations)
    elif mode == 'train':
        worker = pilot_gpu if accelerator == 'gpu' else pilot
        experiment = Path(experiment_config).read_text(encoding='utf-8') if experiment_config else ''
        result = worker.remote(seed, warm_steps, updates, batch, eval_pairs, wall_seconds, resume, init_from_v2 or None, experiment)
    elif mode == 'poker':
        worker = poker_gpu if accelerator == 'gpu' else poker
        result = worker.remote(seed, steps, batch, samples, wall_seconds, lr, entropy)
    elif mode == 'controls':
        if not 1 <= steps <= 100000 or not 1 <= wall_seconds <= 120:
            raise ValueError('controls exceed bounded experiment limits')
        result = controls_gpu.remote(seed, steps, wall_seconds)
    elif mode == 'control-evaluate':
        if not 1 <= wall_seconds <= 300 or not 1 <= eval_pairs <= 10000:
            raise ValueError('control evaluation exceeds bounded experiment limits')
        result = control_evaluate_gpu.remote(seed, eval_pairs, wall_seconds)
    elif mode == 'evaluate':
        if not candidate.startswith('/artifacts/runs/') or not output.startswith('/artifacts/runs/'):
            raise ValueError('evaluation artifact paths must be inside the runs volume')
        result = evaluate_gpu.remote(candidate, output, eval_pairs, seed, wall_seconds, domain, samples, resume)
    else:
        raise ValueError('mode must be smoke, train, poker, evaluate, controls, or control-evaluate')
    print(json.dumps(result, indent=2))

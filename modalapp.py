from pathlib import Path
import modal

app = modal.App('fly-poker-training')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'numpy==2.3.3', 'pokerkit==0.7.6')
    .add_local_dir(str(Path(__file__).parent / 'training'), remote_path='/root/training', copy=True))
volume = modal.Volume.from_name('fly-poker-artifacts', create_if_missing=True)


@app.function(image=image, cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def smoke(seed: int = 7, batch: int = 8, iterations: int = 3):
    import torch
    from training.data import fetch
    from training.benchmark import benchmark
    torch.manual_seed(seed)
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    return benchmark(str(path), batch=batch, iterations=iterations, device='cpu')


@app.function(image=image, cpu=4, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def pilot(seed: int = 7, warm_steps: int = 64, updates: int = 32, batch: int = 8,
          eval_pairs: int = 15, wall_seconds: int = 900, resume: bool = False):
    from training.budget import Budget
    from training.data import fetch, load_graph
    from training.train import TrainConfig, train
    if not 1 <= wall_seconds <= 900:
        raise ValueError('wall_seconds must be between 1 and 900')
    budget = Budget(wall_seconds)
    config = TrainConfig(seed, warm_steps, updates, batch, eval_pairs, resume)
    config.validate()
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    try:
        return train(load_graph(path), Path(f'/artifacts/runs/seed-{seed}'), config,
            budget, device='cpu', commit=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(mode: str = 'smoke', seed: int = 7, batch: int = 8, iterations: int = 3,
         warm_steps: int = 64, updates: int = 32, eval_pairs: int = 15,
         wall_seconds: int = 900, resume: bool = False):
    import json
    if mode == 'smoke':
        result = smoke.remote(seed, batch, iterations)
    elif mode == 'train':
        result = pilot.remote(seed, warm_steps, updates, batch, eval_pairs, wall_seconds, resume)
    else:
        raise ValueError('mode must be smoke or train')
    print(json.dumps(result, indent=2))

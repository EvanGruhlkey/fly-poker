from pathlib import Path
import modal

app = modal.App('fly-poker-training')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'numpy==2.3.3', 'pokerkit==0.7.6')
    .add_local_dir(str(Path(__file__).parent / 'training'), remote_path='/root/training', copy=True))
volume = modal.Volume.from_name('fly-poker-artifacts', create_if_missing=True)


@app.function(image=image, gpu='L4', cpu=1, memory=8192, max_containers=1, timeout=1200,
              retries=0, volumes={'/artifacts': volume})
def smoke(seed: int = 7, batch: int = 8, iterations: int = 3):
    import torch
    from training.data import fetch
    from training.benchmark import benchmark
    torch.manual_seed(seed)
    path = fetch(Path('/artifacts/data/full.npz'))
    volume.commit()
    return benchmark(str(path), batch=batch, iterations=iterations, device='cuda')


@app.local_entrypoint()
def main(mode: str = 'smoke', seed: int = 7, batch: int = 8, iterations: int = 3,
         warm_steps: int = 32, updates: int = 32, eval_pairs: int = 15, wall_seconds: int = 900):
    import json
    if mode != 'smoke':
        raise ValueError('training entry point is being added; use smoke for the real-graph benchmark')
    print(json.dumps(smoke.remote(seed, batch, iterations), indent=2))

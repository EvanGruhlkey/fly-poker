import time
from pathlib import Path
import torch
from .data import fetch, load_graph, tiny_graph
from .model import BrainPolicy
from .features import FEATURE_COUNT


def benchmark(graph_path: str | None = None, *, batch: int = 8, iterations: int = 3, device: str = 'cpu') -> dict:
    if not 1 <= batch <= 32 or not 1 <= iterations <= 5:
        raise ValueError('benchmark is limited to five iterations and batch 32')
    start = time.monotonic()
    torch.set_num_threads(4)
    graph = load_graph(Path(graph_path)) if graph_path else tiny_graph()
    model = BrainPolicy(graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    features = torch.randn(batch, FEATURE_COUNT, device=device)
    mask = torch.ones(batch, 7, dtype=torch.bool, device=device)
    edges_before = model.syn_gain.detach().clone()
    leak_before = model.leak_logit.detach().clone()
    before = model.sensory.weight.detach().clone()
    times = []
    for _ in range(iterations):
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        step_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(features, mask)
        loss = -torch.log_softmax(logits, -1)[:, 4].mean() + values.square().mean()
        loss.backward()
        optimizer.step()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        times.append(time.monotonic() - step_start)
    return {**model.metadata(), 'device': device, 'batch': batch, 'iterations': iterations,
        'step_seconds': times, 'total_seconds': time.monotonic() - start,
        'gpu_name': torch.cuda.get_device_name() if device.startswith('cuda') else None,
        'edge_magnitudes_changed': not torch.equal(edges_before, model.syn_gain),
        'neuron_leaks_changed': not torch.equal(leak_before, model.leak_logit),
        'fixed_signs_preserved': torch.equal(model.edge_weights().sign().to(torch.int8), model.signs),
        'gradients_finite': all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()),
        'sensory_weights_changed': not torch.equal(before, model.sensory.weight),
        'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        'synthetic': graph_path is None}


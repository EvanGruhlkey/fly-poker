from dataclasses import dataclass

import torch
from torch import Tensor

from .encoding import SpikeBatch
from .loader import SparseConnectome


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    duration_ms: int = 150
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("simulation duration must be positive")
        if self.threshold <= 0:
            raise ValueError("spike threshold must be positive")


@dataclass(frozen=True, slots=True)
class ActivityBatch:
    spike_counts: Tensor
    device: str


def simulate(
    connectome: SparseConnectome,
    spikes: SpikeBatch,
    config: SimulatorConfig,
    device: str = "cpu",
) -> ActivityBatch:
    """Run fixed-weight synchronous spiking propagation."""
    source = torch.as_tensor(connectome.source, dtype=torch.long, device=device)
    target = torch.as_tensor(connectome.target, dtype=torch.long, device=device)
    weight = torch.as_tensor(connectome.weight, dtype=torch.float32, device=device)
    batch_size = spikes.spikes.shape[0]
    neuron_count = connectome.neuron_count
    fired = torch.zeros((batch_size, neuron_count), dtype=torch.bool, device=device)
    sensory_ids = torch.as_tensor(spikes.neuron_ids, dtype=torch.long, device=device)
    fired[:, sensory_ids] = torch.as_tensor(spikes.spikes, dtype=torch.bool, device=device)
    counts = torch.zeros((batch_size, neuron_count), dtype=torch.int32, device=device)

    with torch.no_grad():
        for _ in range(config.duration_ms):
            counts += fired
            signal = fired[:, source].to(torch.float32) * weight
            potential = torch.zeros_like(fired, dtype=torch.float32)
            potential.index_add_(1, target, signal)
            fired = potential >= config.threshold
    return ActivityBatch(counts, torch.device(device).type)

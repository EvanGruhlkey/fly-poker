import numpy as np
import torch

from fly_poker.connectome.ablation import shuffled_wiring, silenced
from fly_poker.connectome.encoding import SpikeBatch
from fly_poker.connectome.loader import SparseConnectome
from fly_poker.connectome.simulator import SimulatorConfig, simulate


def graph() -> SparseConnectome:
    return SparseConnectome(
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([2.0, 2.0], dtype=np.float32),
        np.array(["sensory", "central", "motor"]),
    )


def test_spikes_propagate_through_tiny_graph_in_batches() -> None:
    spikes = SpikeBatch(np.array([0]), np.array([[True], [False]]))

    activity = simulate(graph(), spikes, SimulatorConfig(duration_ms=3, threshold=1.0))

    assert activity.spike_counts.tolist() == [[1, 1, 1], [0, 0, 0]]
    assert activity.device == "cpu"


def test_connectome_tensors_are_fixed() -> None:
    spikes = SpikeBatch(np.array([0]), np.array([[True]]))

    activity = simulate(graph(), spikes, SimulatorConfig(duration_ms=3))

    assert not activity.spike_counts.requires_grad
    assert activity.spike_counts.grad_fn is None


def test_ablations_are_deterministic_and_do_not_mutate_source() -> None:
    original = graph()
    first = shuffled_wiring(original, seed=4)
    second = shuffled_wiring(original, seed=4)

    assert np.array_equal(first.target, second.target)
    assert np.array_equal(original.target, [1, 2])
    assert np.all(silenced(original).weight == 0)


def test_cpu_and_cuda_match_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        return
    spikes = SpikeBatch(np.array([0]), np.array([[True]]))
    config = SimulatorConfig(duration_ms=3)

    cpu = simulate(graph(), spikes, config, device="cpu")
    cuda = simulate(graph(), spikes, config, device="cuda")

    assert torch.equal(cpu.spike_counts, cuda.spike_counts.cpu())

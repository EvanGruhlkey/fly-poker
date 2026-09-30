import numpy as np

from .loader import SparseConnectome


def silenced(connectome: SparseConnectome) -> SparseConnectome:
    return SparseConnectome(
        connectome.source.copy(),
        connectome.target.copy(),
        np.zeros_like(connectome.weight),
        connectome.neuron_class.copy(),
    )


def shuffled_wiring(connectome: SparseConnectome, seed: int) -> SparseConnectome:
    rng = np.random.default_rng(seed)
    target = connectome.target.copy()
    rng.shuffle(target)
    return SparseConnectome(
        connectome.source.copy(),
        target,
        connectome.weight.copy(),
        connectome.neuron_class.copy(),
    )

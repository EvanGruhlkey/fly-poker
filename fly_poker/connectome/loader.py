from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .manifest import ConnectomeManifest


@dataclass(frozen=True, slots=True)
class SparseConnectome:
    source: NDArray[np.int32]
    target: NDArray[np.int32]
    weight: NDArray[np.float32]
    neuron_class: NDArray[np.str_]

    @property
    def neuron_count(self) -> int:
        return len(self.neuron_class)

    @property
    def edge_count(self) -> int:
        return len(self.source)

    @property
    def sensory_ids(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.neuron_class == "sensory")

    @property
    def motor_ids(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.neuron_class == "motor")


def load_connectome(path: Path, manifest: ConnectomeManifest) -> SparseConnectome:
    if not path.exists():
        command = "uv run python -m fly_poker.connectome.fetch"
        raise FileNotFoundError(f"MaleCNS data missing; run `{command}`")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != manifest.sha256:
        raise ValueError("connectome checksum does not match manifest")

    with np.load(path, mmap_mode="r", allow_pickle=False) as data:
        graph = SparseConnectome(
            data["source"].astype(np.int32, copy=False),
            data["target"].astype(np.int32, copy=False),
            data["weight"].astype(np.float32, copy=False),
            data["neuron_class"].astype(np.str_, copy=False),
        )
    if graph.neuron_count != manifest.neuron_count:
        raise ValueError("connectome neuron count does not match manifest")
    if graph.edge_count != manifest.edge_count:
        raise ValueError("connectome edge count does not match manifest")
    if not (len(graph.target) == len(graph.weight) == graph.edge_count):
        raise ValueError("connectome edge arrays have different lengths")
    if graph.edge_count and max(graph.source.max(), graph.target.max()) >= graph.neuron_count:
        raise ValueError("connectome edge references an unknown neuron")
    return graph

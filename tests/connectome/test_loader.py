from hashlib import sha256

import numpy as np
import pytest

from fly_poker.connectome.loader import load_connectome
from fly_poker.connectome.manifest import ConnectomeManifest


def write_graph(path) -> str:
    np.savez(
        path,
        source=np.array([0, 1], dtype=np.int32),
        target=np.array([1, 2], dtype=np.int32),
        weight=np.array([4.0, -2.0], dtype=np.float32),
        neuron_class=np.array(["sensory", "central", "motor"]),
    )
    return sha256(path.read_bytes()).hexdigest()


def test_loads_checksum_verified_sparse_graph(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    digest = write_graph(path)
    manifest = ConnectomeManifest("fixture", digest, 3, 2)

    graph = load_connectome(path, manifest)

    assert graph.neuron_count == 3
    assert graph.edge_count == 2
    assert graph.sensory_ids.tolist() == [0]
    assert graph.motor_ids.tolist() == [2]


def test_rejects_corrupt_graph(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    write_graph(path)
    manifest = ConnectomeManifest("fixture", "0" * 64, 3, 2)

    with pytest.raises(ValueError, match="checksum"):
        load_connectome(path, manifest)


def test_rejects_manifest_shape_mismatch(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    digest = write_graph(path)
    manifest = ConnectomeManifest("fixture", digest, 4, 2)

    with pytest.raises(ValueError, match="neuron count"):
        load_connectome(path, manifest)


def test_missing_data_names_acquisition_command(tmp_path) -> None:
    manifest = ConnectomeManifest("fixture", "0" * 64, 3, 2)

    with pytest.raises(FileNotFoundError, match="uv run python -m fly_poker.connectome.fetch"):
        load_connectome(tmp_path / "missing.npz", manifest)

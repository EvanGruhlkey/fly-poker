import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
import numpy as np

MANIFEST = json.loads(Path(__file__).with_name('data_manifest.json').read_text(encoding='utf-8'))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, manifest: dict = MANIFEST):
    if path.stat().st_size != manifest['bytes'] or file_sha(path) != manifest['sha256']:
        raise ValueError('connectome checksum or size does not match the pinned manifest')


def fetch(path: Path, manifest: dict = MANIFEST) -> Path:
    path = Path(path)
    if path.exists():
        try:
            verify(path, manifest)
            return path
        except ValueError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(manifest['url'], timeout=30) as source:
                size = 0
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > manifest['bytes']:
                        raise ValueError('download exceeds pinned dataset size')
                    temporary.write(block)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        verify(temporary_path, manifest)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class Graph:
    root_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    counts: np.ndarray
    signs: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    sha256: str
    meta: dict

    @property
    def n(self) -> int:
        return len(self.root_ids)

    def validate(self):
        n, edges = self.n, len(self.indices)
        if self.indptr.shape != (n + 1,) or self.indptr[0] != 0 or self.indptr[-1] != edges or np.any(np.diff(self.indptr) < 0):
            raise ValueError('invalid CSR row pointers')
        if len(np.unique(self.root_ids)) != n or self.counts.shape != (edges,) or self.signs.shape != (edges,):
            raise ValueError('invalid neuron or edge arrays')
        if not np.isfinite(self.counts).all() or np.any(self.counts <= 0) or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError('invalid synaptic count or transmitter sign')
        for indices in (self.indices, self.inputs, self.outputs):
            if indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= n):
                raise ValueError('invalid neuron indices')
        if not len(self.inputs) or not len(self.outputs) or len(np.unique(self.inputs)) != len(self.inputs) or len(np.unique(self.outputs)) != len(self.outputs):
            raise ValueError('invalid sensory or motor neurons')
        rows = np.repeat(np.arange(n), np.diff(self.indptr))
        if np.any(rows == self.indices):
            raise ValueError('self loops are not allowed')
        positive = np.bincount(self.indices, weights=self.signs > 0, minlength=n)
        negative = np.bincount(self.indices, weights=self.signs < 0, minlength=n)
        if np.any((positive > 0) & (negative > 0)):
            raise ValueError('presynaptic transmitter signs are inconsistent')


def load_graph(path: Path) -> Graph:
    verify(path)
    with np.load(path, allow_pickle=False) as archive:
        graph = Graph(archive['root_ids'], archive['csr_indptr'], archive['csr_indices'],
            archive['syn_count'], archive['sign'], archive['input_idx'], archive['output_idx'],
            MANIFEST['sha256'], json.loads(str(archive['meta'])))
        if archive['super_class'].shape != (graph.n,):
            raise ValueError('invalid neuron classification array')
    graph.validate()
    if graph.n != MANIFEST['neurons'] or len(graph.indices) != MANIFEST['connections'] or len(graph.inputs) != MANIFEST['inputs'] or len(graph.outputs) != MANIFEST['outputs']:
        raise ValueError('dataset dimensions differ from pinned manifest')
    return graph


def tiny_graph() -> Graph:
    graph = Graph(np.arange(5, dtype=np.int64), np.array([0, 0, 0, 2, 4, 5]),
        np.array([0, 1, 1, 2, 3]), np.array([2, 1, 2, 3, 1], dtype=np.float32),
        np.array([1, -1, -1, 1, 1], dtype=np.int8), np.array([0, 1]), np.array([3, 4]),
        'synthetic-test-only', {'synthetic': True})
    graph.validate()
    return graph

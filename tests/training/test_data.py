import hashlib
from pathlib import Path
import pytest
from training.data import fetch, tiny_graph


def test_fetch_checksums_and_atomic_replace(tmp_path):
    source = tmp_path / 'source.npz'
    source.write_bytes(b'verified data')
    target = tmp_path / 'cache' / 'graph.npz'
    manifest = {'url': source.as_uri(), 'bytes': source.stat().st_size, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    assert fetch(target, manifest).read_bytes() == b'verified data'
    bad = dict(manifest, sha256='0' * 64)
    with pytest.raises(ValueError):
        fetch(target, bad)
    assert target.read_bytes() == b'verified data'
    assert not list(target.parent.glob('*.partial'))


def test_tiny_graph_is_explicitly_not_real_connectome():
    graph = tiny_graph()
    assert graph.n == 5 and graph.sha256 == 'synthetic-test-only'
    assert graph.meta['synthetic']

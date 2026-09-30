import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConnectomeManifest:
    version: str
    sha256: str
    neuron_count: int
    edge_count: int

    @classmethod
    def read(cls, path: Path) -> "ConnectomeManifest":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

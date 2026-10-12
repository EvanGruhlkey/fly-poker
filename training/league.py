from copy import deepcopy
import os
from pathlib import Path
import tempfile
import torch
from .checkpoint import FIXED_BUFFERS
from .model import BrainPolicy


class OpponentLeague:
    def __init__(self, model: BrainPolicy, directory: Path, interval: int = 32,
                 capacity: int = 3, *, update: int = 0):
        if interval < 1 or capacity < 1 or update < 0:
            raise ValueError('invalid opponent league configuration')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.capacity = capacity
        self._snapshots: dict[int, BrainPolicy] = {}
        paths = {}
        for path in self.directory.glob('snapshot_*.pt'):
            suffix = path.stem.removeprefix('snapshot_')
            if suffix.isdigit() and int(suffix) <= update:
                paths[int(suffix)] = path
        for saved_update in sorted(paths)[-capacity:]:
            payload = torch.load(paths[saved_update], map_location=next(model.parameters()).device,
                weights_only=True)
            if payload['metadata'] != model.metadata() or payload['update'] != saved_update:
                raise ValueError('opponent snapshot identity mismatch')
            state = payload['model_state']
            if any(not torch.equal(state[key], model.state_dict()[key]) for key in FIXED_BUFFERS):
                raise ValueError('opponent snapshot modifies fixed connectome buffers')
            snapshot = self._frozen_copy(model)
            snapshot.load_state_dict(state, strict=True)
            self._snapshots[saved_update] = snapshot
        if not self._snapshots:
            if update:
                raise ValueError('opponent snapshots missing for resumed update')
            self._store(model, 0)

    @staticmethod
    def _frozen_copy(model: BrainPolicy) -> BrainPolicy:
        snapshot = deepcopy(model)
        snapshot.eval()
        snapshot.requires_grad_(False)
        return snapshot

    def _store(self, model: BrainPolicy, update: int):
        snapshot = self._frozen_copy(model)
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save({'metadata': model.metadata(), 'update': update,
                    'model_state': snapshot.state_dict()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, self.directory / f'snapshot_{update}.pt')
        finally:
            temporary.unlink(missing_ok=True)
        self._snapshots[update] = snapshot
        self._snapshots = dict(sorted(self._snapshots.items())[-self.capacity:])

    def refresh(self, model: BrainPolicy, update: int) -> bool:
        if update <= 0 or update % self.interval or update in self._snapshots:
            return False
        if update < max(self._snapshots):
            raise ValueError('opponent league updates must increase')
        self._store(model, update)
        return True

    def policies(self) -> dict[str, BrainPolicy]:
        return {f'snapshot_{update}': model for update, model in self._snapshots.items()}

    def metadata(self) -> dict:
        return {'interval': self.interval, 'capacity': self.capacity,
            'updates': list(self._snapshots)}

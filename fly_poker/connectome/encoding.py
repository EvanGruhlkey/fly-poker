import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from fly_poker.poker.actions import PokerAction
from fly_poker.poker.engine import HandSnapshot


CHANNELS = (
    "street",
    "position",
    "pot_odds",
    "effective_stack",
    "hole_cards",
    "board_cards",
    "action_kind",
    "action_size",
    "revision",
)


@dataclass(frozen=True, slots=True)
class EncoderMetadata:
    version: str
    channels: tuple[str, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class SpikeBatch:
    neuron_ids: NDArray[np.int64]
    spikes: NDArray[np.bool_]


ENCODER_METADATA = EncoderMetadata(
    "poker-sensory-v1",
    CHANNELS,
    sha256("\n".join(CHANNELS).encode()).hexdigest(),
)


def encode_candidates(
    snapshot: HandSnapshot,
    actions: tuple[PokerAction, ...],
    sensory_ids: NDArray[np.integer],
    seed: int,
) -> SpikeBatch:
    """Map each candidate to stable sensory-neuron spikes."""
    ids = np.asarray(sensory_ids, dtype=np.int64)
    spikes = np.empty((len(actions), len(ids)), dtype=np.bool_)
    for row, action in enumerate(actions):
        payload = _payload(snapshot, action, seed)
        for column, neuron_id in enumerate(ids):
            digest = sha256(payload + int(neuron_id).to_bytes(8, "little", signed=True)).digest()
            spikes[row, column] = int.from_bytes(digest[:2], "little") < 16384
    return SpikeBatch(ids, spikes)


def _payload(snapshot: HandSnapshot, action: PokerAction, seed: int) -> bytes:
    actor = snapshot.actor if snapshot.actor is not None else 0
    actor_hole = sorted(snapshot.hole_cards[actor])
    to_call = next(
        (candidate.amount for candidate in snapshot.legal_actions if candidate.kind.value == "call"),
        0,
    )
    effective = min(snapshot.stacks)
    features = {
        "version": ENCODER_METADATA.version,
        "seed": seed,
        "street": snapshot.street,
        "position": actor,
        "pot_odds": round(to_call / max(1, snapshot.pot + to_call), 6),
        "effective_stack": round(effective / max(1, snapshot.pot), 6),
        "hole_cards": actor_hole,
        "board_cards": sorted(snapshot.board),
        "action_kind": action.kind.value,
        "action_size": round(action.amount / max(1, snapshot.pot), 6),
        "revision": snapshot.revision,
    }
    return json.dumps(features, sort_keys=True, separators=(",", ":")).encode()

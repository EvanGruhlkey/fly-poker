from dataclasses import replace

import numpy as np

from fly_poker.connectome.encoding import ENCODER_METADATA, encode_candidates
from fly_poker.poker.engine import HandEngine


def test_same_state_seed_and_mapping_produce_same_spikes() -> None:
    snapshot = HandEngine.new(seed=20).snapshot
    sensory = np.array([19, 3, 11, 7], dtype=np.int64)

    first = encode_candidates(snapshot, snapshot.legal_actions, sensory, seed=5)
    second = encode_candidates(snapshot, snapshot.legal_actions, sensory, seed=5)

    assert np.array_equal(first.spikes, second.spikes)
    assert ENCODER_METADATA.checksum


def test_card_order_is_canonical() -> None:
    snapshot = HandEngine.new(seed=21).snapshot
    reversed_hole = tuple(tuple(reversed(cards)) for cards in snapshot.hole_cards)
    equivalent = replace(snapshot, hole_cards=reversed_hole)
    sensory = np.arange(32)

    first = encode_candidates(snapshot, snapshot.legal_actions, sensory, seed=8)
    second = encode_candidates(equivalent, equivalent.legal_actions, sensory, seed=8)

    assert np.array_equal(first.spikes, second.spikes)


def test_actions_and_position_change_stimulation() -> None:
    snapshot = HandEngine.new(seed=22).snapshot
    sensory = np.arange(128)

    batch = encode_candidates(snapshot, snapshot.legal_actions, sensory, seed=9)
    other_actor = replace(snapshot, actor=1)
    other = encode_candidates(other_actor, other_actor.legal_actions, sensory, seed=9)

    assert not np.array_equal(batch.spikes[0], batch.spikes[-1])
    assert not np.array_equal(batch.spikes, other.spikes)


def test_mapping_is_stable_when_neuron_ids_are_reordered() -> None:
    snapshot = HandEngine.new(seed=23).snapshot
    forward = np.array([2, 5, 9, 14, 21])
    reverse = forward[::-1]

    a = encode_candidates(snapshot, snapshot.legal_actions, forward, seed=10)
    b = encode_candidates(snapshot, snapshot.legal_actions, reverse, seed=10)

    assert np.array_equal(a.spikes, b.spikes[:, ::-1])

import pytest

from fly_poker.poker.actions import ActionKind
from fly_poker.poker.engine import HandEngine, StaleRevision


def choose(engine: HandEngine, kind: ActionKind) -> None:
    action = next(item for item in engine.snapshot.legal_actions if item.kind is kind)
    engine.apply(action, engine.snapshot.revision)


def test_seeded_hand_posts_blinds_and_deals_repeatably() -> None:
    first = HandEngine.new(seed=42, stack_bb=100)
    second = HandEngine.new(seed=42, stack_bb=100)

    assert first.snapshot == second.snapshot
    assert first.snapshot.pot == 150
    assert first.snapshot.stacks == (9950, 9900)
    assert len(first.snapshot.hole_cards[0]) == 2
    assert first.snapshot.actor == 0


def test_fold_ends_hand_and_awards_the_pot() -> None:
    engine = HandEngine.new(seed=7, stack_bb=100)

    choose(engine, ActionKind.FOLD)

    assert engine.snapshot.terminal
    assert engine.snapshot.stacks == (9950, 10050)
    assert engine.events[-1].kind == "fold"


def test_checks_advance_each_street_to_showdown() -> None:
    engine = HandEngine.new(seed=8, stack_bb=100)
    choose(engine, ActionKind.CALL)
    choose(engine, ActionKind.CHECK)

    assert engine.snapshot.street == "flop"
    assert len(engine.snapshot.board) == 3

    for expected_street in ("turn", "river"):
        choose(engine, ActionKind.CHECK)
        choose(engine, ActionKind.CHECK)
        assert engine.snapshot.street == expected_street

    choose(engine, ActionKind.CHECK)
    choose(engine, ActionKind.CHECK)
    assert engine.snapshot.terminal
    assert engine.events[-1].kind == "showdown"


def test_all_in_call_runs_out_board_without_side_pot() -> None:
    engine = HandEngine.new(seed=9, stack_bb=10)
    choose(engine, ActionKind.ALL_IN)
    choose(engine, ActionKind.CALL)

    assert engine.snapshot.terminal
    assert len(engine.snapshot.board) == 5
    assert sum(engine.snapshot.stacks) == 2000


def test_stale_revision_cannot_advance_hand_twice() -> None:
    engine = HandEngine.new(seed=10, stack_bb=100)
    action = engine.snapshot.legal_actions[0]
    engine.apply(action, expected_revision=0)

    with pytest.raises(StaleRevision):
        engine.apply(action, expected_revision=0)

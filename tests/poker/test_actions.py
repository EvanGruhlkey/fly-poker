from fly_poker.poker.actions import ActionKind, legal_abstract_actions
from fly_poker.poker.state import PublicHandState


def amounts(state: PublicHandState) -> list[tuple[ActionKind, int]]:
    return [(action.kind, action.amount) for action in legal_abstract_actions(state)]


def test_facing_bet_offers_fold_call_and_raises() -> None:
    state = PublicHandState(pot=100, to_call=20, stack=300, min_raise=60)

    assert amounts(state) == [
        (ActionKind.FOLD, 0),
        (ActionKind.CALL, 20),
        (ActionKind.HALF_POT, 80),
        (ActionKind.POT, 140),
        (ActionKind.DOUBLE_POT, 260),
        (ActionKind.ALL_IN, 300),
    ]


def test_unopened_action_offers_check_and_bets() -> None:
    state = PublicHandState(pot=100, to_call=0, stack=300, min_raise=20)

    assert amounts(state) == [
        (ActionKind.CHECK, 0),
        (ActionKind.HALF_POT, 50),
        (ActionKind.POT, 100),
        (ActionKind.DOUBLE_POT, 200),
        (ActionKind.ALL_IN, 300),
    ]


def test_short_stack_collapses_duplicate_sizes_to_all_in() -> None:
    state = PublicHandState(pot=100, to_call=20, stack=80, min_raise=60)

    assert amounts(state) == [
        (ActionKind.FOLD, 0),
        (ActionKind.CALL, 20),
        (ActionKind.ALL_IN, 80),
    ]


def test_all_in_call_does_not_offer_raise() -> None:
    state = PublicHandState(pot=100, to_call=80, stack=80, min_raise=120)

    assert amounts(state) == [
        (ActionKind.FOLD, 0),
        (ActionKind.CALL, 80),
    ]

from dataclasses import dataclass
from enum import StrEnum

from .state import PublicHandState


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    HALF_POT = "half_pot"
    POT = "pot"
    DOUBLE_POT = "double_pot"
    ALL_IN = "all_in"


@dataclass(frozen=True, slots=True)
class PokerAction:
    kind: ActionKind
    amount: int = 0


def legal_abstract_actions(state: PublicHandState) -> tuple[PokerAction, ...]:
    """Return distinct legal actions; amount is chips paid on this action."""
    actions: list[PokerAction] = []
    if state.to_call:
        actions.extend(
            (PokerAction(ActionKind.FOLD), PokerAction(ActionKind.CALL, state.to_call))
        )
    else:
        actions.append(PokerAction(ActionKind.CHECK))

    if not state.can_raise or state.stack <= state.to_call:
        return tuple(actions)

    pot_after_call = state.pot + state.to_call
    candidates = (
        (ActionKind.HALF_POT, state.to_call + pot_after_call // 2),
        (ActionKind.POT, state.to_call + pot_after_call),
        (ActionKind.DOUBLE_POT, state.to_call + pot_after_call * 2),
    )
    seen = {action.amount for action in actions}
    for kind, amount in candidates:
        if amount < state.min_raise or amount >= state.stack or amount in seen:
            continue
        actions.append(PokerAction(kind, amount))
        seen.add(amount)

    actions.append(PokerAction(ActionKind.ALL_IN, state.stack))
    return tuple(actions)

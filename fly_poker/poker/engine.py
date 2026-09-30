from dataclasses import dataclass
from random import Random

from pokerkit import StandardHighHand

from .actions import ActionKind, PokerAction, legal_abstract_actions
from .events import HandEvent
from .state import PublicHandState


class StaleRevision(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HandSnapshot:
    revision: int
    street: str
    actor: int | None
    pot: int
    stacks: tuple[int, int]
    board: tuple[str, ...]
    hole_cards: tuple[tuple[str, ...], tuple[str, ...]]
    legal_actions: tuple[PokerAction, ...]
    terminal: bool


class HandEngine:
    def __init__(self, seed: int, stack_bb: int) -> None:
        rng = Random(seed)
        self._deck = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
        rng.shuffle(self._deck)
        self._holes = (tuple(self._draw(2)), tuple(self._draw(2)))
        starting = stack_bb * 100
        self._stacks = [starting - 50, starting - 100]
        self._committed = [50, 100]
        self._pot = 150
        self._board: list[str] = []
        self._street = "preflop"
        self._actor: int | None = 0
        self._terminal = False
        self._revision = 0
        self._last_raise = 100
        self._actions_since_raise = 0
        self.events = [
            HandEvent("blind", 0, 50),
            HandEvent("blind", 1, 100),
            HandEvent("deal"),
        ]

    @classmethod
    def new(cls, seed: int, stack_bb: int = 100) -> "HandEngine":
        if stack_bb < 2:
            raise ValueError("stack must be at least two big blinds")
        return cls(seed, stack_bb)

    @property
    def snapshot(self) -> HandSnapshot:
        return HandSnapshot(
            self._revision,
            self._street,
            self._actor,
            self._pot,
            tuple(self._stacks),
            tuple(self._board),
            self._holes,
            self._legal_actions(),
            self._terminal,
        )

    def apply(self, action: PokerAction, expected_revision: int) -> HandSnapshot:
        if expected_revision != self._revision:
            raise StaleRevision(f"expected revision {self._revision}")
        if action not in self._legal_actions() or self._actor is None:
            raise ValueError("illegal action")
        actor = self._actor
        if action.kind is ActionKind.FOLD:
            self._stacks[1 - actor] += self._pot
            self._pot = 0
            self._revision += 1
            self._finish(HandEvent("fold", actor))
            return self.snapshot

        if action.amount:
            self._stacks[actor] -= action.amount
            self._committed[actor] += action.amount
            self._pot += action.amount
        aggressive = action.kind in {
            ActionKind.HALF_POT,
            ActionKind.POT,
            ActionKind.DOUBLE_POT,
            ActionKind.ALL_IN,
        }
        self.events.append(HandEvent(action.kind.value, actor, action.amount))
        if aggressive:
            previous_high = self._committed[1 - actor]
            self._last_raise = max(1, self._committed[actor] - previous_high)
            self._actions_since_raise = 1
        else:
            self._actions_since_raise += 1
        self._revision += 1
        self._actor = 1 - actor

        equal = self._committed[0] == self._committed[1]
        if equal and 0 in self._stacks:
            self._runout_and_showdown()
        elif equal and self._actions_since_raise >= 2:
            self._advance_street()
        return self.snapshot

    def _legal_actions(self) -> tuple[PokerAction, ...]:
        if self._terminal or self._actor is None:
            return ()
        actor = self._actor
        to_call = max(self._committed) - self._committed[actor]
        state = PublicHandState(
            pot=self._pot,
            to_call=min(to_call, self._stacks[actor]),
            stack=self._stacks[actor],
            min_raise=to_call + self._last_raise,
            can_raise=self._stacks[1 - actor] > 0,
        )
        return legal_abstract_actions(state)

    def _advance_street(self) -> None:
        self._committed = [0, 0]
        self._actions_since_raise = 0
        self._last_raise = 100
        if self._street == "preflop":
            self._street, count = "flop", 3
        elif self._street == "flop":
            self._street, count = "turn", 1
        elif self._street == "turn":
            self._street, count = "river", 1
        else:
            self._showdown()
            return
        cards = tuple(self._draw(count))
        self._board.extend(cards)
        self.events.append(HandEvent(self._street, cards=cards))
        self._actor = 1

    def _runout_and_showdown(self) -> None:
        if len(self._board) < 5:
            cards = tuple(self._draw(5 - len(self._board)))
            self._board.extend(cards)
            self.events.append(HandEvent("runout", cards=cards))
        self._showdown()

    def _showdown(self) -> None:
        board = "".join(self._board)
        hands = [StandardHighHand.from_game("".join(hole), board) for hole in self._holes]
        if hands[0] == hands[1]:
            half = self._pot // 2
            self._stacks[0] += half + self._pot % 2
            self._stacks[1] += half
        else:
            winner = 0 if hands[0] > hands[1] else 1
            self._stacks[winner] += self._pot
        self._pot = 0
        self._finish(HandEvent("showdown"))

    def _finish(self, event: HandEvent) -> None:
        self.events.append(event)
        self._terminal = True
        self._actor = None

    def _draw(self, count: int) -> list[str]:
        return [self._deck.pop() for _ in range(count)]

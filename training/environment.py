from dataclasses import dataclass
from random import Random
from pokerkit import Automation, NoLimitTexasHoldem

ACTION_VERSION = 'holdem-seven-v1'
ACTION_NAMES = ('fold', 'check_call', 'min_raise', 'half_pot', 'pot', 'double_pot', 'all_in')
AUTOMATIONS = (
    Automation.ANTE_POSTING, Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING, Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
)


@dataclass(frozen=True)
class Observation:
    hole: tuple[str, str]
    board: tuple[str, ...]
    street: int
    dealer: bool
    pot: int
    to_call: int
    stack: int
    opponent_stack: int
    committed: int
    opponent_committed: int
    actions: tuple[tuple[str, int] | None, ...]

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(action is not None for action in self.actions)


class Holdem:
    def __init__(self, seed: int, dealer: int = 0):
        if dealer not in (0, 1):
            raise ValueError('dealer must be 0 or 1')
        self.dealer = dealer
        self._seats = (1 - dealer, dealer)
        self._deck = [rank + suit for rank in '23456789TJQKA' for suit in 'cdhs']
        Random(seed).shuffle(self._deck)
        self._cursor = 0
        self._state = NoLimitTexasHoldem.create_state(AUTOMATIONS, False, 0, (2, 4), 4, (400, 400), 2)
        while self._state.can_deal_hole():
            self._state.deal_hole(self._draw(1))
        self._advance()

    def _draw(self, count: int) -> str:
        cards = ''.join(self._deck[self._cursor:self._cursor + count])
        self._cursor += count
        return cards

    def _advance(self):
        while self._state.status and self._state.actor_index is None:
            if self._state.can_burn_card():
                self._state.burn_card(self._draw(1))
            elif self._state.can_deal_board():
                self._state.deal_board(self._draw(self._state.board_dealing_count))
            else:
                raise RuntimeError('PokerKit stopped outside a betting or terminal state')

    @property
    def terminal(self) -> bool:
        return not self._state.status

    @property
    def actor(self) -> int | None:
        index = self._state.actor_index
        return None if index is None else self._seats[index]

    @property
    def stacks(self) -> tuple[int, int]:
        stacks = self._state.stacks
        return (stacks[self._seats.index(0)], stacks[self._seats.index(1)])

    @property
    def pot(self) -> int:
        return self._state.total_pot_amount

    def actions(self) -> tuple[tuple[str, int] | None, ...]:
        if self.terminal:
            return (None,) * 7
        state = self._state
        actor = state.actor_index
        call = state.checking_or_calling_amount
        actions: list[tuple[str, int] | None] = [None] * 7
        if call and state.can_fold():
            actions[0] = ('fold', 0)
        if state.can_check_or_call():
            actions[1] = ('check_call', call)
        if state.can_complete_bet_or_raise_to():
            minimum = state.min_completion_betting_or_raising_to_amount
            maximum = state.max_completion_betting_or_raising_to_amount
            matched = state.bets[actor] + call
            targets = (minimum, matched + max(4, round((self.pot + call) * .5)),
                       matched + max(4, self.pot + call), matched + max(4, 2 * (self.pot + call)), maximum)
            seen = set(action for action in actions if action is not None)
            for index, target in enumerate(targets, 2):
                target = min(maximum, max(minimum, target))
                action = ('raise', target)
                if action not in seen and state.can_complete_bet_or_raise_to(target):
                    actions[index] = action
                    seen.add(action)
        return tuple(actions)

    def observe(self) -> Observation:
        if self.terminal:
            raise ValueError('terminal hands have no private observation')
        state = self._state
        actor = state.actor_index
        return Observation(tuple(str(card) for card in state.hole_cards[actor]),
            tuple(str(cards[0]) for cards in state.board_cards), state.street_index,
            self.actor == self.dealer, self.pot, state.checking_or_calling_amount,
            state.stacks[actor], state.stacks[1 - actor], state.bets[actor], state.bets[1 - actor], self.actions())

    def step(self, index: int):
        actions = self.actions()
        if not isinstance(index, int) or not 0 <= index < 7 or actions[index] is None:
            raise ValueError('illegal abstract action')
        kind, amount = actions[index]
        if kind == 'fold':
            self._state.fold()
        elif kind == 'check_call':
            self._state.check_or_call()
        else:
            self._state.complete_bet_or_raise_to(amount)
        self._advance()

    def reward(self, seat: int) -> float:
        if not self.terminal:
            raise ValueError('reward is defined only after final settlement')
        return (self.stacks[seat] - 400) / 4

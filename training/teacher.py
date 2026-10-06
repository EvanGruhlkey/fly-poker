import numpy as np
from pokerkit import StandardHighHand
from .budget import Budget
from .environment import Observation, Holdem

DECK = tuple(rank + suit for rank in '23456789TJQKA' for suit in 'cdhs')


def equity(obs: Observation, rng: np.random.Generator, samples: int = 16, budget: Budget | None = None) -> float:
    known = set(obs.hole + obs.board)
    unseen = tuple(card for card in DECK if card not in known)
    wins = 0.0
    for _ in range(samples):
        if budget:
            budget.check()
        drawn = rng.choice(unseen, size=2 + 5 - len(obs.board), replace=False).tolist()
        board = ''.join(obs.board + tuple(drawn[2:]))
        own = StandardHighHand.from_game(''.join(obs.hole), board)
        opponent = StandardHighHand.from_game(''.join(drawn[:2]), board)
        wins += 1 if own > opponent else .5 if own == opponent else 0
    return wins / samples


def heuristic(obs: Observation, win_probability: float) -> int:
    odds = obs.to_call / max(1, obs.pot + obs.to_call)
    if obs.mask[0] and win_probability + .05 < odds:
        return 0
    if win_probability > .76:
        for index in (4, 3, 2, 6, 5):
            if obs.mask[index]:
                return index
    elif win_probability > .62 and obs.to_call == 0:
        for index in (3, 2):
            if obs.mask[index]:
                return index
    return 1


def baseline(obs: Observation, kind: str, rng: np.random.Generator, budget: Budget) -> int:
    if kind == 'random':
        return int(rng.choice(np.flatnonzero(obs.mask)))
    if kind == 'check_call':
        return 1
    if kind == 'equity':
        return heuristic(obs, equity(obs, rng, samples=8, budget=budget))
    raise ValueError('unknown baseline')


def sample_states(rng: np.random.Generator, count: int, budget: Budget) -> list[Observation]:
    states: list[Observation] = []
    while len(states) < count:
        budget.check()
        env = Holdem(int(rng.integers(0, 2**31)), dealer=int(rng.integers(0, 2)))
        hand = []
        while not env.terminal:
            budget.check()
            observation = env.observe()
            hand.append(observation)
            action = 1 if rng.random() < .8 else int(rng.choice(np.flatnonzero(observation.mask)))
            env.step(action)
        rng.shuffle(hand)
        states.extend(hand)
    return states[:count]

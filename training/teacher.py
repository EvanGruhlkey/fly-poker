import numpy as np
from pokerkit import StandardHighHand
from .budget import Budget
from .environment import Observation, Holdem

DECK = tuple(rank + suit for rank in '23456789TJQKA' for suit in 'cdhs')


def equity(obs: Observation, rng: np.random.Generator, samples: int = 64, budget: Budget | None = None) -> float:
    known = set(obs.hole + obs.board)
    unseen = tuple(card for card in DECK if card not in known)
    wins = 0.0
    total_weight = 0.0
    for _ in range(samples):
        if budget:
            budget.check()
        drawn = rng.choice(unseen, size=2 + 5 - len(obs.board), replace=False).tolist()
        board = ''.join(obs.board + tuple(drawn[2:]))
        own = StandardHighHand.from_game(''.join(obs.hole), board)
        opponent = StandardHighHand.from_game(''.join(drawn[:2]), board)
        weight = range_weight(obs, tuple(drawn[:2]))
        wins += weight * (1 if own > opponent else .5 if own == opponent else 0)
        total_weight += weight
    return wins / total_weight


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


def range_weight(obs: Observation, hole: tuple[str, str]) -> float:
    ranks = sorted(('23456789TJQKA'.index(card[0]) + 2 for card in hole), reverse=True)
    strength = (sum(ranks) / 28 + .5 * (ranks[0] == ranks[1]) + .1 * (hole[0][1] == hole[1][1])) / 1.6
    pressure = sum(min(1, event.paid / 80) for event in obs.history if not event.actor and event.kind == 'raise')
    return float(np.exp(min(3, pressure) * 2 * strength))


def action_targets(obs: Observation, probability: float) -> tuple[np.ndarray, float]:
    values = np.full(7, -np.inf)
    for index, action in enumerate(obs.actions):
        if action is None:
            continue
        kind, amount = action
        if kind == 'fold':
            values[index] = 0
            continue
        paid = max(0, amount - obs.committed) if kind == 'raise' else amount
        response = min(obs.opponent_stack, max(0, obs.committed + paid - obs.opponent_committed)) if kind == 'raise' else 0
        matched = min(paid, max(0, obs.opponent_committed + obs.opponent_stack - obs.committed))
        contested_pot = obs.pot - max(0, obs.opponent_committed - (obs.committed + matched))
        values[index] = probability * (contested_pot + response) - (1 - probability) * matched
    temperature = max(4, .2 * obs.pot)
    weights = np.exp((values - np.max(values)) / temperature)
    targets = weights / weights.sum()
    return targets.astype(np.float32), float(np.clip(np.dot(targets[np.array(obs.mask)], values[np.array(obs.mask)]) / 800, -1, 1))


def initial_stacks(rng: np.random.Generator, *, variable: bool) -> tuple[int, int]:
    buyin = int(rng.integers(4, 797)) if variable and rng.random() < .25 else 400
    return buyin, 800 - buyin


def sample_states(rng: np.random.Generator, count: int, budget: Budget) -> list[Observation]:
    states: list[Observation] = []
    while len(states) < count:
        budget.check()
        env = Holdem(int(rng.integers(0, 2**30)), dealer=int(rng.integers(0, 2)), stacks=initial_stacks(rng, variable=False))
        hand = []
        while not env.terminal:
            budget.check()
            observation = env.observe()
            hand.append(observation)
            action = 1 if rng.random() < .8 else int(rng.choice(np.flatnonzero(observation.mask)))
            env.step(action)
        wanted_street = len(states) % 4
        matching = [obs for obs in hand if obs.street == wanted_street]
        if matching:
            states.append(matching[int(rng.integers(len(matching)))])
    return states

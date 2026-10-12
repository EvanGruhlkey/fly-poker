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
    if kind == 'teacher':
        probabilities, _ = poker_targets(obs, rng, samples=32, budget=budget)
        return int(rng.choice(7, p=probabilities / probabilities.sum()))
    raise ValueError('unknown baseline')


def range_weight(obs: Observation, hole: tuple[str, str]) -> float:
    ranks = sorted(('23456789TJQKA'.index(card[0]) + 2 for card in hole), reverse=True)
    strength = (sum(ranks) / 28 + .5 * (ranks[0] == ranks[1]) + .1 * (hole[0][1] == hole[1][1])) / 1.6
    pressure = sum(min(1, event.paid / 80) for event in obs.history if not event.actor and event.kind == 'raise')
    return float(np.exp(min(3, pressure) * 2 * strength))


def hand_strength(hole: tuple[str, str]) -> float:
    ranks = sorted(('23456789TJQKA'.index(card[0]) + 2 for card in hole), reverse=True)
    return float((sum(ranks) / 28 + .5 * (ranks[0] == ranks[1]) + .1 * (hole[0][1] == hole[1][1])) / 1.6)


def chip_terms(obs: Observation, kind: str, amount: int) -> tuple[int, int, int, int]:
    paid = max(0, amount - obs.committed) if kind == 'raise' else amount
    response = min(obs.opponent_stack, max(0, obs.committed + paid - obs.opponent_committed)) if kind == 'raise' else 0
    matched = min(paid, max(0, obs.opponent_committed + obs.opponent_stack - obs.committed))
    contested = obs.pot - max(0, obs.opponent_committed - (obs.committed + matched))
    return paid, response, matched, contested


def distribute(obs: Observation, values: np.ndarray) -> tuple[np.ndarray, float]:
    temperature = max(4, .2 * obs.pot)
    finite = np.isfinite(values)
    weights = np.exp((values - np.max(values)) / temperature)
    weights[~finite] = 0
    targets = weights / weights.sum()
    chosen = values.copy()
    chosen[~finite] = 0
    return targets.astype(np.float32), float(np.clip(np.dot(targets, chosen) / 800, -1, 1))


def action_targets(obs: Observation, probability: float) -> tuple[np.ndarray, float]:
    values = np.full(7, -np.inf)
    for index, action in enumerate(obs.actions):
        if action is None:
            continue
        kind, amount = action
        if kind == 'fold':
            values[index] = 0
            continue
        _, response, matched, contested = chip_terms(obs, kind, amount)
        values[index] = probability * (contested + response) - (1 - probability) * matched
    return distribute(obs, values)


def heuristic_targets(obs: Observation, probability: float) -> tuple[np.ndarray, float]:
    legal = np.asarray(obs.mask, dtype=bool)
    probabilities = legal.astype(np.float32) * (.02 / legal.sum())
    probabilities[heuristic(obs, probability)] += .98
    values = np.zeros(7)
    for index, action in enumerate(obs.actions):
        if action is not None and action[0] != 'fold':
            _, response, matched, contested = chip_terms(obs, *action)
            values[index] = probability * (contested + response) - (1 - probability) * matched
    net = float(np.dot(probabilities, values)) - (obs.starting_stack - obs.stack)
    return probabilities, net / 800


def poker_values(obs: Observation, samples: list[tuple[float, float, float]]) -> np.ndarray:
    if not samples or any(weight <= 0 or not 0 <= win <= 1 or not 0 <= strength <= 1 for weight, win, strength in samples):
        raise ValueError('showdown samples must be positive weights and unit-interval outcomes')
    weights = np.array([sample[0] for sample in samples], dtype=np.float64)
    wins = np.array([sample[1] for sample in samples], dtype=np.float64)
    strength = np.array([sample[2] for sample in samples], dtype=np.float64)
    total = float(weights.sum())
    equity_all = float(np.clip(np.dot(weights, wins) / total, 0., 1.))
    values = np.full(7, -np.inf)
    for index, action in enumerate(obs.actions):
        if action is None:
            continue
        kind, amount = action
        if kind == 'fold':
            values[index] = 0
            continue
        paid, response, matched, contested = chip_terms(obs, kind, amount)
        if kind != 'raise' or response <= 0:
            values[index] = equity_all * (contested + response) - (1 - equity_all) * matched
            continue
        required = response / max(1, obs.pot + paid + response)
        calling = strength >= required
        call_weight = float(weights[calling].sum())
        if call_weight == 0:
            values[index] = contested
            continue
        equity_called = float(np.clip(np.dot(weights[calling], wins[calling]) / call_weight, 0., 1.))
        called = equity_called * (contested + response) - (1 - equity_called) * matched
        fold_rate = float(np.clip(1 - call_weight / total, 0., 1.))
        effective_samples = total * total / float(np.dot(weights, weights))
        uncertainty = .5 * (contested + response + matched) * np.sqrt(
            equity_all * (1 - equity_all) / effective_samples)
        values[index] = fold_rate * contested + (1 - fold_rate) * called - uncertainty
    return values


def decision_values(obs: Observation, samples: list[tuple[float, float, float]]) -> tuple[np.ndarray, float]:
    return distribute(obs, poker_values(obs, samples))


def action_rewards(obs: Observation, samples: list[tuple[float, float, float]]) -> np.ndarray:
    values = poker_values(obs, samples)
    rewards = np.zeros(7, dtype=np.float32)
    finite = np.isfinite(values)
    rewards[finite] = np.clip(values[finite] / 800, -1, 1)
    return rewards


def sample_showdowns(obs: Observation, rng: np.random.Generator, samples: int = 32, budget: Budget | None = None) -> list[tuple[float, float, float]]:
    if samples < 1:
        raise ValueError('positive showdown sample count required')
    known = set(obs.hole + obs.board)
    unseen = tuple(card for card in DECK if card not in known)
    rows = []
    for _ in range(samples):
        if budget:
            budget.check()
        drawn = rng.choice(unseen, size=2 + 5 - len(obs.board), replace=False).tolist()
        board = ''.join(obs.board + tuple(drawn[2:]))
        own = StandardHighHand.from_game(''.join(obs.hole), board)
        opponent = StandardHighHand.from_game(''.join(drawn[:2]), board)
        hole = tuple(drawn[:2])
        win = 1.0 if own > opponent else .5 if own == opponent else 0.0
        rows.append((range_weight(obs, hole), win, opponent_equity(obs, hole, rng, samples=8, budget=budget)))
    return rows


def opponent_equity(obs: Observation, hole: tuple[str, str], rng: np.random.Generator,
                    samples: int = 8, budget: Budget | None = None) -> float:
    known = set(hole + obs.board)
    unseen = tuple(card for card in DECK if card not in known)
    wins = 0.
    for _ in range(samples):
        if budget:
            budget.check()
        drawn = rng.choice(unseen, 2 + 5 - len(obs.board), replace=False).tolist()
        board = ''.join(obs.board + tuple(drawn[2:]))
        own = StandardHighHand.from_game(''.join(hole), board)
        other = StandardHighHand.from_game(''.join(drawn[:2]), board)
        wins += 1 if own > other else .5 if own == other else 0
    return wins / samples


def net_targets(obs: Observation, samples: list[tuple[float, float, float]]) -> tuple[np.ndarray, float]:
    values = poker_values(obs, samples)
    probabilities, _ = distribute(obs, values)
    incremental = float(np.dot(probabilities, np.where(np.isfinite(values), values, 0)))
    invested = obs.starting_stack - obs.stack
    return probabilities, (incremental - invested) / 800


def poker_targets(obs: Observation, rng: np.random.Generator, samples: int = 32, budget: Budget | None = None) -> tuple[np.ndarray, float]:
    return decision_values(obs, sample_showdowns(obs, rng, samples, budget))


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

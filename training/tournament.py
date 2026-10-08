from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .budget import Budget, BudgetExceeded
from .environment import Holdem, Observation
from .teacher import equity, heuristic

Policy = Callable[[Observation], np.ndarray]


@dataclass(frozen=True)
class TournamentConfig:
    pairs: int = 1000
    seed: int = 9182
    stack_profiles: tuple[tuple[int, int], ...] = ((400, 400), (80, 720), (200, 600))
    domain: str = 'validation'
    match_hands: int = 1

    def __post_init__(self):
        if self.domain not in ('validation', 'test'):
            raise ValueError('evaluation domain must be validation or test')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError('seed must be an unsigned 64-bit integer')
        if not isinstance(self.pairs, int) or not 1 <= self.pairs < 2**64:
            raise ValueError('pairs must be positive')
        if not isinstance(self.match_hands, int) or not 1 <= self.match_hands < 2**32:
            raise ValueError('match_hands must be positive')
        if not self.stack_profiles or len(set(self.stack_profiles)) != len(self.stack_profiles):
            raise ValueError('stack profiles must be nonempty and unique')
        for profile in self.stack_profiles:
            if len(profile) != 2 or any(not isinstance(x, int) or x <= 0 for x in profile) or sum(profile) != 800:
                raise ValueError('stack profiles must conserve 800 positive integer chips')

    @property
    def deal_seeds(self) -> list[int]:
        domain = 1 if self.domain == 'validation' else 2
        return [-(domain * 2**160 + self.seed * 2**96 + i * 2**32 + 1)
                for i in range(self.pairs)]


def summarize(samples: list[float]) -> dict:
    count = len(samples)
    interval = None
    if count >= 2:
        rng = np.random.default_rng(9182)
        means = np.array([rng.choice(samples, count, replace=True).mean() for _ in range(5000)])
        interval = np.quantile(means, [.025, .975]).tolist()
    return {'mean_bb_per_hand': float(np.mean(samples)) if count else None,
            'ci95': interval, 'paired_samples': count,
            'ci95_method': 'paired sample percentile bootstrap; 5000 resamples'}


def baseline_policy(kind: str, *, samples: int = 8, budget: Budget | None = None) -> Policy:
    if kind not in ('random', 'passive', 'aggressive', 'equity'):
        raise ValueError('unknown baseline')

    def policy(obs):
        legal = np.flatnonzero(obs.mask)
        probabilities = np.zeros(7)
        if kind == 'random':
            probabilities[legal] = 1 / len(legal)
        else:
            index = 1
            if kind == 'aggressive':
                index = next((i for i in (4, 3, 2, 6, 5, 1, 0) if obs.mask[i]))
            elif kind == 'equity':
                seed = np.random.SeedSequence([ord(c) for c in ''.join(obs.hole + obs.board)] +
                                             [obs.pot, obs.to_call, obs.stack])
                index = heuristic(obs, equity(obs, np.random.default_rng(seed), samples, budget))
            probabilities[index] = 1
        return probabilities
    return policy


def _probabilities(policy: Policy, obs: Observation) -> np.ndarray:
    values = np.asarray(policy(obs), dtype=float)
    if (values.shape != (7,) or not np.isfinite(values).all() or (values < 0).any()
            or (values[~np.array(obs.mask)] != 0).any() or not np.isclose(values.sum(), 1)):
        raise ValueError('policy must return seven normalized legal probabilities')
    return values / values.sum()


def _match(policy, opponent, seed, seat, stacks, hands, budget):
    rng = np.random.default_rng(abs(seed))
    initial = stacks[seat]
    counts = {}
    completed = 0
    for hand in range(hands):
        budget.check()
        if min(stacks) == 0:
            break
        env = Holdem(seed - hand, dealer=hand % 2, stacks=stacks)
        while not env.terminal:
            budget.check()
            actor = env.actor
            obs = env.observe()
            probabilities = _probabilities(policy if actor == seat else opponent, obs)
            action = int(rng.choice(7, p=probabilities))
            if actor == seat:
                key = f'street_{obs.street}_action_{action}'
                counts[key] = counts.get(key, 0) + 1
            env.step(action)
        stacks = env.stacks
        completed += 1
    return (stacks[seat] - initial) / 4 / completed, counts, completed


def tournament(candidate: Policy, starting: Policy, budget: Budget,
               config: TournamentConfig = TournamentConfig(), *, opponents=None) -> dict:
    opponents = opponents if opponents is not None else {
        name: baseline_policy(name, budget=budget) for name in ('random', 'passive', 'equity', 'aggressive')}
    if not opponents:
        raise ValueError('opponent panel must be nonempty')
    profiles = {}
    exhausted = False
    for stacks in config.stack_profiles:
        profile = {}
        for rival, opponent in opponents.items():
            returns = {label: [] for label in ('candidate', 'starting')}
            counts = {label: {} for label in returns}
            played = {label: 0 for label in returns}
            try:
                for seed in config.deal_seeds:
                    pair = {}
                    for label, policy in (('candidate', candidate), ('starting', starting)):
                        outcomes = [_match(policy, opponent, seed, seat,
                                          stacks if seat == 0 else stacks[::-1], config.match_hands, budget)
                                    for seat in (0, 1)]
                        pair[label] = sum(result[0] for result in outcomes) / 2
                        for _, actions, hands in outcomes:
                            played[label] += hands
                            for action, count in actions.items():
                                counts[label][action] = counts[label].get(action, 0) + count
                    for label in returns:
                        returns[label].append(pair[label])
            except BudgetExceeded:
                exhausted = True
            for label, samples in returns.items():
                profile[f'{label}_vs_{rival}'] = {
                    **summarize(samples), 'pair_returns_bb': samples, 'action_counts': counts[label],
                    'hands': played[label], 'requested_pairs': config.pairs,
                    'complete': len(samples) == config.pairs}
            gain = [a - b for a, b in zip(returns['candidate'], returns['starting'], strict=True)]
            profile[f'paired_gain_vs_{rival}'] = {**summarize(gain), 'pair_returns_bb': gain,
                                                 'complete': len(gain) == config.pairs}
        profiles[f'{stacks[0]}_{stacks[1]}'] = profile
    return {'config': asdict(config), 'deal_seeds': config.deal_seeds, 'profiles': profiles,
            'complete': all(entry['complete'] for p in profiles.values() for entry in p.values()),
            'budget_exhausted': exhausted, 'policy_mode': 'sampled probabilities',
            'sample_unit': 'seat-swapped match pair' if config.match_hands > 1 else 'seat-swapped hand pair'}


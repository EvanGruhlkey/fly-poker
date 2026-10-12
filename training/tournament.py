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
        values = np.asarray(samples, dtype=float)
        means = np.empty(5000)
        batch = max(1, min(128, 1_000_000 // count))
        for offset in range(0, len(means), batch):
            size = min(batch, len(means) - offset)
            means[offset:offset + size] = values[rng.integers(count, size=(size, count))].mean(axis=1)
        interval = np.quantile(means, [.025, .975]).tolist()
    return {'mean_bb_per_hand': float(np.mean(samples)) if count else None,
            'ci95': interval, 'paired_samples': count,
            'bb_per_100': float(np.mean(samples) * 100) if count else None,
            'ci95_bb_per_100': [value * 100 for value in interval] if interval else None,
            'ci95_method': 'paired sample percentile bootstrap; 5000 resamples'}


def baseline_policy(kind: str, *, samples: int = 8, budget: Budget | None = None) -> Policy:
    if kind not in ('random', 'passive', 'aggressive', 'equity', 'teacher'):
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
            elif kind in ('equity', 'teacher'):
                seed = np.random.SeedSequence([ord(c) for c in ''.join(obs.hole + obs.board)] +
                                             [obs.pot, obs.to_call, obs.stack])
                if kind == 'teacher':
                    from .teacher import poker_targets
                    return poker_targets(obs, np.random.default_rng(seed), samples=samples, budget=budget)[0]
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
        if action != 0:
            counts['showdowns'] = counts.get('showdowns', 0) + 1
            if env.stacks[seat] > stacks[seat]:
                counts['showdown_wins'] = counts.get('showdown_wins', 0) + 1
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
            seats = {label: {'0': [], '1': []} for label in returns}
            try:
                for seed in config.deal_seeds:
                    pair = {}
                    pair_outcomes = {}
                    for label, policy in (('candidate', candidate), ('starting', starting)):
                        outcomes = [_match(policy, opponent, seed, seat,
                                          stacks if seat == 0 else stacks[::-1], config.match_hands, budget)
                                    for seat in (0, 1)]
                        pair[label] = sum(result[0] for result in outcomes) / 2
                        pair_outcomes[label] = outcomes
                    for label in returns:
                        returns[label].append(pair[label])
                        outcomes = pair_outcomes[label]
                        for seat, (_, actions, hands) in enumerate(outcomes):
                            seats[label][str(seat)].append(outcomes[seat][0])
                            played[label] += hands
                            for action, count in actions.items():
                                counts[label][action] = counts[label].get(action, 0) + count

            except BudgetExceeded:
                exhausted = True
            for label, samples in returns.items():
                action_totals = {str(action): sum(value for key, value in counts[label].items()
                                                  if key.endswith(f'_action_{action}')) for action in range(7)}
                total = sum(action_totals.values())
                profile[f'{label}_vs_{rival}'] = {
                    'showdown_frequency': counts[label].get('showdowns', 0) / played[label] if played[label] else None,
                    'showdown_win_frequency': counts[label].get('showdown_wins', 0) / counts[label]['showdowns'] if counts[label].get('showdowns') else None,
                    'seats': {seat: summarize(values) for seat, values in seats[label].items()},
                    'action_frequencies': {key: value / total if total else None for key, value in action_totals.items()},
                    'aggression_frequency': sum(action_totals[str(i)] for i in range(2, 7)) / total if total else None,
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



def promotion(report: dict, *, parity: bool) -> dict:
    reasons = []
    if not parity:
        reasons.append('export/schema parity not verified')
    if not report.get('complete'):
        reasons.append('evaluation incomplete')
    config = report.get('config', {})
    if config.get('domain') != 'test' or config.get('pairs', 0) < 1000:
        reasons.append('promotion requires final test domain with at least 1000 paired decks')
    profiles = report.get('profiles', {})
    if not {'400_400', '80_720', '200_600'}.issubset(profiles):
        reasons.append('promotion requires all three stack profiles')
    for name, profile in profiles.items():
        required = {'random', 'passive', 'aggressive', 'equity', 'teacher', 'v2'}
        panel = required | {key.removeprefix('candidate_vs_') for key in profile if key.startswith('candidate_vs_')}
        for rival in sorted(panel):
            for prefix in ('candidate', 'paired_gain'):
                gate = f'{prefix}_vs_{rival}'
                entry = profile.get(gate, {})
                interval = entry.get('ci95')
                if (not entry.get('complete') or entry.get('paired_samples', 0) < 1000
                        or interval is None or len(interval) != 2
                        or not np.isfinite(interval).all() or interval[0] <= 0):
                    reasons.append(f'{name}: {gate} requires 1000 pairs and a positive lower confidence bound')
    return {'promoted': not reasons, 'reasons': reasons}


def compare_training_seeds(reports: dict[int, dict]) -> dict:
    if not reports:
        raise ValueError('at least one training-seed report required')
    first = next(iter(reports.values()))
    for report in reports.values():
        if report['config'] != first['config'] or report['deal_seeds'] != first['deal_seeds']:
            raise ValueError('training seeds must use identical evaluation manifests')
        if report['profiles'].keys() != first['profiles'].keys():
            raise ValueError('training seeds must use identical stack profiles')
        for name, profile in report['profiles'].items():
            if profile.keys() != first['profiles'][name].keys():
                raise ValueError('training seeds must use identical opponent panels')
    profiles = {}
    for name, profile in first['profiles'].items():
        profiles[name] = {}
        for key in profile:
            entries = [report['profiles'][name][key] for report in reports.values()]
            means = [entry['mean_bb_per_hand'] for entry in entries if entry['mean_bb_per_hand'] is not None]
            profiles[name][key] = {**summarize(means), 'per_seed_means': means,
                                  'complete': all(entry['complete'] for entry in entries)}
    return {'training_seeds': list(reports), 'profiles': profiles, 'config': first['config'],
            'deal_seeds': first['deal_seeds'], 'sample_unit': 'training seed',
            'complete': all(report['complete'] for report in reports.values())}


def torch_policy(model, *, observation_features: int | None = None) -> Policy:
    import torch
    from .features import encode
    observation_features = observation_features or model.config.observation_features
    device = next(model.parameters()).device
    model.eval()

    def policy(obs):
        features = torch.as_tensor(encode(obs)[:observation_features], device=device).unsqueeze(0)
        mask = torch.tensor([obs.mask], dtype=torch.bool, device=device)
        with torch.no_grad():
            output = model(features, mask)
            logits = output[0] if isinstance(output, tuple) else output
            return torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return policy

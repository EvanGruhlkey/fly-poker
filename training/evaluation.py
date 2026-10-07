import numpy as np
from .budget import Budget, BudgetExceeded
from .model import BrainPolicy
from .rollout import rollout, EVAL_SEED_BASE


def summarize(pair_means: list[float]) -> dict:
    count = len(pair_means)
    mean = float(np.mean(pair_means)) if count else None
    interval = None
    if count >= 2:
        rng = np.random.default_rng(9182)
        resampled = rng.choice(pair_means, size=(5000, count), replace=True).mean(axis=1)
        interval = np.quantile(resampled, [.025, .975]).tolist()
    return {'mean_bb_per_hand': mean, 'ci95': interval, 'ci95_method': 'paired-deck percentile bootstrap; 5000 resamples', 'sample_limited': True, 'paired_decks': count, 'hands': count * 2,
        'conclusion': 'inconclusive' if interval is None or interval[0] <= 0 <= interval[1] else 'limited-sample positive' if mean > 0 else 'limited-sample negative'}


def evaluate(model: BrainPolicy, starting: BrainPolicy, budget: Budget, *, seed: int, pairs: int) -> dict:
    reports = {}
    for opponent in ('random', 'equity'):
        candidate_returns = {}
        for label, candidate in (('trained', model), ('starting', starting)):
            returns = []
            exhausted = False
            frequencies = {}
            try:
                for pair_start in range(0, pairs, 4):
                    budget.check()
                    count = min(4, pairs - pair_start)
                    seeds = [EVAL_SEED_BASE + (seed % 100000) * 1000 + pair_start + i for i in range(count) for _ in range(2)]
                    seats = [seat for _ in range(count) for seat in (0, 1)]
                    rng = np.random.default_rng(EVAL_SEED_BASE + seed + pair_start)
                    episodes = rollout(candidate, starting, rng, budget, seeds=seeds, learner_seats=seats, opponents=[opponent] * len(seeds))
                    returns.extend((episodes[i].reward_bb + episodes[i + 1].reward_bb) / 2 for i in range(0, len(episodes), 2))
                    for episode in episodes:
                        for observation, action in zip(episode.observations, episode.actions, strict=True):
                            key = f'street_{observation.street}_action_{action}'
                            frequencies[key] = frequencies.get(key, 0) + 1
            except BudgetExceeded:
                exhausted = True
            candidate_returns[label] = returns
            reports[f'{label}_vs_{opponent}'] = {**summarize(returns), 'opponent': opponent,
                'requested_pairs': pairs, 'complete': len(returns) == pairs, 'budget_exhausted': exhausted,
                'policy_mode': 'sampled-softmax-temperature-1', 'action_counts': frequencies, 'pair_returns_bb': returns}
        overlap = min(map(len, candidate_returns.values()))
        differences = [candidate_returns['trained'][i] - candidate_returns['starting'][i] for i in range(overlap)]
        reports[f'paired_gain_vs_{opponent}'] = {**summarize(differences), 'complete': overlap == pairs,
            'budget_exhausted': overlap != pairs, 'interpretation': 'Matched 100-BB decks, swapped seats, and per-hand policy random seeds; sampled new minus sampled starting policy.'}
    return reports

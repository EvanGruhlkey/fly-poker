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


def evaluate(model: BrainPolicy, untrained: BrainPolicy, budget: Budget, *, seed: int, pairs: int) -> dict:
    specifications = [('trained_vs_random', model, 'random', 'real'),
        ('trained_vs_check_call', model, 'check_call', 'real'),
        ('trained_vs_equity', model, 'equity', 'real'),
        ('zero_edges_vs_random', model, 'random', 'zero'),
        ('shuffled_wiring_vs_random', model, 'random', 'shuffle'),
        ('untrained_vs_random', untrained, 'random', 'real')]
    reports = {}
    for name, candidate, opponent, ablation in specifications:
        returns = []
        exhausted = False
        for pair_start in range(0, pairs, 4):
            try:
                budget.check()
                count = min(4, pairs - pair_start)
                seeds = [EVAL_SEED_BASE + (seed % 100000) * 1000 + pair_start + i for i in range(count) for _ in range(2)]
                seats = [seat for _ in range(count) for seat in (0, 1)]
                rng = np.random.default_rng(EVAL_SEED_BASE + seed + pair_start)
                episodes = rollout(candidate, untrained, rng, budget, seeds=seeds, learner_seats=seats,
                    opponents=[opponent] * len(seeds), greedy=True, ablation=ablation)
                returns.extend((episodes[i].reward_bb + episodes[i + 1].reward_bb) / 2 for i in range(0, len(episodes), 2))
            except BudgetExceeded:
                exhausted = True
                break
        reports[name] = {**summarize(returns), 'graph_intervention': ablation,
            'opponent': opponent, 'requested_pairs': pairs, 'complete': len(returns) == pairs,
            'budget_exhausted': exhausted, 'policy_mode': 'greedy', 'pair_returns_bb': returns}
    return reports

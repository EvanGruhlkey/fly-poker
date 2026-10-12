import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .budget import Budget, BudgetExceeded
from .kuhn import FEATURE_COUNT, HISTORIES, Kuhn, assess
from .nfsp import NFSPAgent, masked_probabilities
from .nfsp_rollout import collect_hand


def average_strategy(agents):
    table = {}
    for card in range(3):
        for history in HISTORIES:
            seat = len(history) % 2
            features = np.zeros(FEATURE_COUNT, dtype=np.float32)
            features[card] = features[3 + seat] = features[5 + HISTORIES.index(history)] = 1
            with torch.no_grad():
                probabilities = masked_probabilities(agents[seat].average(torch.from_numpy(features)),
                                                     torch.ones(2, dtype=torch.bool))
            table[(card, history)] = probabilities.numpy().astype(float)
    return table


def run(*, seed=17, hands=2000, seconds=30, hidden=32, batch=32):
    if not 1 <= hands <= 1000000 or not 1 <= hidden <= 256 or not 1 <= batch <= 256 or seconds <= 0:
        raise ValueError('invalid bounded Kuhn experiment configuration')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agents = [NFSPAgent(FEATURE_COUNT, 2, hidden=hidden, capacity=10000) for _ in range(2)]
    initial = average_strategy(agents)
    before = assess(lambda card, history: initial[(card, history)])
    budget = Budget(seconds)
    completed = 0
    exhausted = False
    losses = []
    try:
        for index in range(hands):
            budget.check()
            epsilon = max(.05, .3 * (1 - index / hands))
            collect_hand(Kuhn(int(rng.integers(2**31))), agents, rng, eta=.1, epsilon=epsilon)
            for agent in agents:
                agent.learn(rng, batch=batch)
            completed += 1
            if index % 100 == 0:
                losses.append({'hand': completed, 'q_updates': [agent.updates for agent in agents]})
    except BudgetExceeded:
        exhausted = True
    final = average_strategy(agents)
    after = assess(lambda card, history: final[(card, history)])
    return {'experiment': 'bounded Kuhn NFSP pilot',
            'config': {'seed': seed, 'hands': hands, 'seconds': seconds, 'hidden': hidden,
                       'batch': batch, 'capacity': 10000, 'eta': .1, 'epsilon_initial': .3,
                       'epsilon_floor': .05, 'learning_rate': .001},
            'measurement': 'exact best response to learned average strategy',
            'before': before, 'after': after,
            'exploitability_change': after['exploitability'] - before['exploitability'],
            'improved': bool(after['exploitability'] < before['exploitability']),
            'completed_hands': completed, 'complete': completed == hands,
            'budget_exhausted': exhausted, 'elapsed_seconds': budget.elapsed,
            'average_strategy': {f'{card}:{history}': probabilities.tolist()
                                 for (card, history), probabilities in final.items()},
            'memory_sizes': [{'replay': len(agent.replay.items), 'supervised': len(agent.supervised.items)}
                             for agent in agents], 'progress': losses,
            'scope': 'Kuhn only; no evidence of full no-limit Holdem strength or exploitability'}


def main():
    parser = argparse.ArgumentParser(description='Measure bounded NFSP learning with exact Kuhn best responses')
    parser.add_argument('--output', type=Path, default=Path('experiments/kuhn-nfsp-pilot.json'))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--hands', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    report = run(seed=args.seed, hands=args.hands, seconds=args.seconds, hidden=args.hidden, batch=args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('before', 'after', 'improved', 'completed_hands', 'complete')}, indent=2))


if __name__ == '__main__':
    main()


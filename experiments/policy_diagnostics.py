import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from training.batched_evaluation import load_model
from training.budget import Budget, BudgetExceeded
from training.data import load_graph
from training.dataset import load_dataset
from training.environment import Holdem
from training.features import encode


ACTION_NAMES = ('fold', 'check_call', 'min_raise', 'half_pot', 'pot', 'double_pot', 'all_in')


def diagnostics(probabilities, targets):
    wanted = targets.argmax(axis=1)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((7, 7), dtype=int)
    np.add.at(confusion, (wanted, predicted), 1)
    fold, raises = wanted == 0, wanted >= 2
    return {'states': len(wanted),
        'cross_entropy': float(-(targets * np.log(np.clip(probabilities, 1e-30, 1))).sum(axis=1).mean()),
        'target_entropy': float(-(targets * np.log(np.clip(targets, 1e-30, 1))).sum(axis=1).mean()),
        'action_accuracy': float((predicted == wanted).mean()),
        'mean_predicted_action_mass': probabilities.mean(axis=0).tolist(),
        'mean_target_action_mass': targets.mean(axis=0).tolist(),
        'confusion_target_rows_predicted_columns': confusion.tolist(),
        'per_action_recall': {name: {'states': int((wanted == i).sum()),
            'recall': float((predicted[wanted == i] == i).mean()) if (wanted == i).any() else None}
            for i, name in enumerate(ACTION_NAMES)},
        'fold_target_states': int(fold.sum()),
        'fold_recall': float((predicted[fold] == 0).mean()) if fold.any() else None,
        'fold_probability_on_fold_targets': float(probabilities[fold, 0].mean()) if fold.any() else None,
        'raise_target_states': int(raises.sum()),
        'raise_category_recall': float((predicted[raises] >= 2).mean()) if raises.any() else None,
        'exact_raise_action_recall': float((predicted[raises] == wanted[raises]).mean()) if raises.any() else None,
        'raise_probability_on_raise_targets': float(probabilities[raises, 2:].sum(axis=1).mean()) if raises.any() else None}


def distributions(model, features, masks):
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(features), torch.as_tensor(masks, dtype=torch.bool))
    return logits.softmax(-1).numpy()


def run(args):
    budget = Budget(args.seconds)
    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.dataset, split='validation')
    budget.check()
    graph = load_graph(args.graph)
    budget.check()
    report = {'action_order': ACTION_NAMES, 'dataset': dataset['metadata'],
              'budget_seconds': args.seconds, 'device': 'cpu', 'models': {}, 'complete': False,
              'measurement': 'heldout teacher imitation and fixed-public-state card sensitivity; no poker strength claim'}
    observations = {label: replace(Holdem(4242).observe(), hole=hole) for label, hole in {
        'AA': ('As', 'Ah'), 'KK': ('Ks', 'Kh'), '72_offsuit': ('7c', '2d'),
        '98_suited': ('9s', '8s')}.items()}
    obs = observations['AA']
    report['card_probe_public_state'] = {'pot': obs.pot, 'to_call': obs.to_call,
        'stack': obs.stack, 'opponent_stack': obs.opponent_stack, 'dealer': obs.dealer,
        'board': obs.board, 'mask': obs.mask, 'actions': obs.actions}
    try:
        for label, path in (('imitation', args.candidate), ('v2', args.starting)):
            budget.check()
            model, artifact = load_model(path, graph, 'cpu')
            probabilities = []
            for start in range(0, len(dataset['values']), args.batch):
                budget.check()
                probabilities.append(distributions(model, dataset['features'][start:start + args.batch],
                                                     dataset['masks'][start:start + args.batch]))
            predicted = np.concatenate(probabilities)
            budget.check()
            probes = distributions(model, np.stack([encode(obs) for obs in observations.values()]),
                                    np.asarray([obs.mask for obs in observations.values()]))
            weak = probes[2]
            report['models'][label] = {'artifact': artifact,
                'validation': diagnostics(predicted, dataset['probabilities']),
                'card_probabilities': {name: row.tolist() for name, row in zip(observations, probes, strict=True)},
                'total_variation_from_72': {name: float(np.abs(row - weak).sum() / 2)
                                          for name, row in zip(observations, probes, strict=True)},
                'card_probe_max_action_mass_range': float(np.ptp(probes, axis=0).max())}
            print(json.dumps({'model': label, **report['models'][label]['validation']}), flush=True)
            del model
        report['complete'] = True
    except BudgetExceeded:
        report['budget_exhausted'] = True
    report['elapsed_seconds'] = budget.elapsed
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Bounded CPU imitation and private-card diagnostics')
    parser.add_argument('--candidate', type=Path, default=Path('runs/round2-imitation.pt'))
    parser.add_argument('--starting', type=Path, default=Path('runs/baseline/v2.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/flywire/full.npz'))
    parser.add_argument('--dataset', type=Path, default=Path('runs/heuristic-validation.npz'))
    parser.add_argument('--output', type=Path, default=Path('experiments/round2-imitation-diagnostics.json'))
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120 or not 1 <= args.batch <= 32 or not 1 <= args.threads <= 4:
        parser.error('diagnostics require 1..120 seconds, batch 1..32 and threads 1..4')
    report = run(args)
    print(json.dumps({'complete': report['complete'], 'elapsed_seconds': report['elapsed_seconds'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()

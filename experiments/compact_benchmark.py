import argparse
import hashlib
import json
from pathlib import Path
import torch
from training.budget import Budget
from training.compact_teacher import loadcompact
from training.teacher_benchmark import validation_gate
from training.tournament import TournamentConfig, baseline_policy, tournament


def benchmark(checkpoint, output, *, pairs=128, samples=16, seconds=180, seed=9182):
    torch.set_num_threads(2)
    budget = Budget(seconds)
    candidate = loadcompact(checkpoint)
    equity = baseline_policy('equity', samples=samples, budget=budget)
    report = tournament(candidate, equity, budget,
        TournamentConfig(pairs=pairs, seed=seed, domain='validation'), opponents={'equity': equity})
    report.update({'candidate': 'compact experimental teacher', 'baseline': 'original equity heuristic',
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'checkpoint': str(checkpoint), 'samples': samples, 'elapsed_seconds': budget.elapsed,
        'production_promoted': False})
    report['validation_gate'] = validation_gate(report)
    report['total_hands'] = sum(entry['hands'] for profile in report['profiles'].values()
        for key, entry in profile.items() if key.startswith(('candidate_vs_', 'starting_vs_')))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in
        ('validation_gate', 'complete', 'total_hands', 'elapsed_seconds')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--seed', type=int, default=9182)
    benchmark(**vars(parser.parse_args()))

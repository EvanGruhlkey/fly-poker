import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import teacher
from .budget import Budget
from .tournament import TournamentConfig, baseline_policy, tournament

TEACHER_VERSION = 'range-conditioned-conservative-showdown-v1'
PROFILES = ('400_400', '80_720', '200_600')


def validation_gate(report: dict) -> dict:
    reasons = []
    config = report.get('config', {})
    if config.get('domain') != 'validation' or config.get('pairs', 0) < 128:
        reasons.append('requires at least 128 validation pairs')
    if not report.get('complete'):
        reasons.append('benchmark incomplete')
    for profile in PROFILES:
        for name in ('candidate_vs_equity', 'paired_gain_vs_equity'):
            entry = report.get('profiles', {}).get(profile, {}).get(name, {})
            interval = entry.get('ci95')
            if (not entry.get('complete') or entry.get('paired_samples', 0) < 128
                    or not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or not np.isfinite(interval).all() or interval[0] <= 0):
                reasons.append(f'{profile}: {name} requires a positive lower confidence bound')
    return {'accepted': not reasons, 'reasons': reasons}


def benchmark(pairs: int = 128, seed: int = 9182, samples: int = 32,
              seconds: float = 600, output: Path | None = None) -> dict:
    if not isinstance(pairs, int) or not 1 <= pairs <= 10000:
        raise ValueError('pairs must be between 1 and 10000')
    if not isinstance(samples, int) or not 1 <= samples <= 4096:
        raise ValueError('samples must be between 1 and 4096')
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    config = TournamentConfig(pairs=pairs, seed=seed, domain='validation')
    budget = Budget(seconds)
    candidate = baseline_policy('teacher', samples=samples, budget=budget)
    baseline = baseline_policy('equity', samples=samples, budget=budget)
    report = tournament(candidate, baseline, budget, config, opponents={'equity': baseline})
    report.update({'teacher_version': TEACHER_VERSION,
        'teacher_sha256': hashlib.sha256(Path(teacher.__file__).read_bytes()).hexdigest(),
        'samples': samples, 'elapsed_seconds': budget.elapsed,
        'observation_rng': 'card characters, pot, call and stack; independent of tournament label',
        'candidate': 'poker_targets', 'baseline': 'original equity heuristic'})
    report['validation_gate'] = validation_gate(report)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate the poker teacher against the equity heuristic.')
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--seed', type=int, default=9182)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2))

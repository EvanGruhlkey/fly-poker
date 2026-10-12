"""Regenerate the measured comparison table from saved tournament reports."""
import json
from pathlib import Path
from training.tournament import summarize

ROOT = Path(__file__).resolve().parent


def number(value):
    return '—' if value is None else f'{value:.2f}'


def interval(value):
    return '—' if value is None else f'[{number(value[0])}, {number(value[1])}]'


def main():
    lines = ['# Round-two measured results', '',
        'Validation experiments only. No new production brain has been promoted.', '',
        'The strength objective remains unmet. Pressure-balanced imitation learned some folding, '
        'but still lost to the equity baseline at every stack size. Its PPO continuation regressed '
        'against equity at equal stacks. The retrained architecture controls show no clear '
        'FlyWire advantage at this sample size. Further scaling of these candidates is not justified.', '',
        'Each sample is a seat-swapped pair on heldout deals. Intervals use the saved paired bootstrap. '
        'Different sample counts have different uncertainty; point estimates alone do not rank models.', '',
        '| Candidate | Stacks (chips) | Opponent | Pairs | BB/100 | 95% CI | Gain over v2 BB/100 | Gain 95% CI |',
        '| --- | --- | --- | ---: | ---: | --- | ---: | --- |']
    notes = []
    for name in ('ppo', 'imitation', 'pressure', 'pressure-imitation'):
        path = ROOT / f'round2-{name}-validation.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for stacks, profile in report['profiles'].items():
            for key, row in profile.items():
                if not key.startswith('candidate_vs_'):
                    continue
                rival = key.removeprefix('candidate_vs_')
                gain = profile[f'paired_gain_vs_{rival}']
                lines.append(f"| {name} | {stacks.replace('_', '/')} | {rival} | "
                    f"{row['paired_samples']} | {number(row['bb_per_100'])} | "
                    f"{interval(row['ci95_bb_per_100'])} | {number(gain['bb_per_100'])} | "
                    f"{interval(gain['ci95_bb_per_100'])} |")
        notes.append(f"{name}: complete panel = {report['complete']}; "
            f"evaluation compute = {report.get('cumulative_elapsed_seconds', report['elapsed_seconds']):.1f} seconds.")
    for note in notes:
        lines.extend(['', note])
    pressure_paths = [ROOT / f'round2-{name}-validation.json'
                      for name in ('pressure', 'pressure-imitation')]
    if all(path.exists() for path in pressure_paths):
        ppo, imitation = [json.loads(path.read_text()) for path in pressure_paths]
        if ppo['config'] != imitation['config']:
            raise ValueError('pressure comparison requires identical evaluation configs')
        lines.extend(['', '## PPO versus matching pressure imitation', '',
            'Same training seed, imitation steps, data, and heldout deals. '
            'Positive values favor PPO. These exploratory intervals do not correct for multiple testing.', '',
            '| Stacks | Opponent | Matched pairs | PPO gain BB/100 | 95% CI |',
            '| --- | --- | ---: | ---: | --- |'])
        for stacks, profile in ppo['profiles'].items():
            for key, row in profile.items():
                if not key.startswith('candidate_vs_'):
                    continue
                baseline = imitation['profiles'][stacks][key]['pair_returns_bb']
                values = row['pair_returns_bb']
                count = min(len(values), len(baseline))
                if not count:
                    continue
                stats = summarize([values[i] - baseline[i] for i in range(count)])
                lines.append(f"| {stacks.replace('_', '/')} | {key.removeprefix('candidate_vs_')} | "
                    f"{count} | {number(stats['bb_per_100'])} | {interval(stats['ci95_bb_per_100'])} |")
    controls_path = ROOT / 'architecture-controls.json'
    control_dir = ROOT / 'controls-validation'
    if controls_path.exists() and all((control_dir / f'{mode}.json').exists()
                                     for mode in ('real', 'shuffle', 'zero', 'mlp')):
        training = json.loads(controls_path.read_text())
        controls = {mode: json.loads((control_dir / f'{mode}.json').read_text())
                    for mode in ('real', 'shuffle', 'zero', 'mlp')}
        lines.extend(['', '## Retrained architecture controls', '',
            'One training seed, about 81 seconds training per mode, identical data-sequence prefixes. '
            'Different step counts reflect throughput under the same wall budget. '
            'These limited controls do not establish biological necessity.', '',
            '| Mode | Updates | Training seconds | Validation loss |',
            '| --- | ---: | ---: | ---: |'])
        for mode, row in training['modes'].items():
            lines.append(f"| {mode} | {row['actual_steps']} | {number(row['training_seconds'])} | "
                         f"{number(row['validation_loss'])} |")
        lines.extend(['', 'Paired FlyWire gain over each control, equally averaging equity/random/v2 '
            'opponents within each heldout deal before bootstrapping. Sample size 32 deals per stack profile. '
            'These exploratory comparisons are not corrected for multiple testing.', '',
            '| Control | Stacks | Real graph gain BB/100 | 95% CI |', '| --- | --- | ---: | --- |'])
        for mode in ('shuffle', 'zero', 'mlp'):
            for stacks, real in controls['real']['profiles'].items():
                rival = controls[mode]['profiles'][stacks]
                differences = [[a - b for a, b in zip(real[f'candidate_vs_{opponent}']['pair_returns_bb'],
                    rival[f'candidate_vs_{opponent}']['pair_returns_bb'], strict=True)]
                    for opponent in ('equity', 'random', 'v2')]
                if any(len(values) != 32 for values in differences):
                    continue
                stats = summarize([sum(values) / 3 for values in zip(*differences, strict=True)])
                lines.append(f"| {mode} | {stacks.replace('_', '/')} | {number(stats['bb_per_100'])} | "
                             f"{interval(stats['ci95_bb_per_100'])} |")
    budget = json.loads((ROOT / 'round2-budget.json').read_text())
    estimate = sum((job['actual_seconds'] if job['actual_seconds'] is not None else job['cap_seconds'])
        / 3600 * budget['rates_usd_per_hour'][job['rate']] for job in budget['jobs'])
    lines.extend(['', f'Compute estimate including capped pending/failed jobs: ${estimate:.2f}. '
        'Startup, storage and billing rounding are excluded. This is not an account invoice.', '',
        '[Training curves](../experiments/round2-learning.png). '
        '[Methods and limitations](poker-strength-audit.md).', ''])
    (ROOT.parent / 'docs' / 'round2-results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote comparison; estimated compute ${estimate:.2f}')


if __name__ == '__main__':
    main()

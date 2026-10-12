"""Plot completed round-two updates, excluding rolled-back log rows."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def completed(name):
    root = Path(__file__).parent
    metrics = json.loads((root / f'round2-{name}-metrics.json').read_text())
    progress = metrics['progress']
    records = {}
    for line in (root / f'round2-{name}-learning.jsonl').read_text().splitlines():
        row = json.loads(line)
        current = row['progress']
        if any(current[key] > progress[key] for key in ('warm_steps', 'updates')):
            continue
        key = (row['phase'], current['warm_steps'], current['updates'])
        records[key] = row
    return list(records.values())


def smooth(values, width=16):
    return [np.mean(values[max(0, index - width + 1):index + 1])
            for index in range(len(values))]


if __name__ == '__main__':
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in ('ppo', 'imitation', 'pressure', 'pressure-imitation'):
        rows = completed(name)
        warm = sorted((row for row in rows if row['phase'] == 'imitation'),
                      key=lambda row: row['progress']['warm_steps'])
        ppo = sorted((row for row in rows if row['phase'] == 'ppo'),
                     key=lambda row: row['progress']['updates'])
        if warm:
            axes[0].plot([row['progress']['warm_steps'] for row in warm],
                         smooth([row['loss'] for row in warm]), label=name)
        if ppo:
            axes[1].plot([row['progress']['updates'] for row in ppo],
                         smooth([row['mean_reward_bb'] for row in ppo]), label=name)
    axes[0].set(xlabel='Completed imitation steps', ylabel='Smoothed training loss')
    axes[1].set(xlabel='Completed PPO updates', ylabel='Smoothed training BB/hand')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle('Round two: training diagnostics, not heldout strength', fontsize=11)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name('round2-learning.png'), dpi=160)

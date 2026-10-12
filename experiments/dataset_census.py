import argparse
import json
from pathlib import Path
import numpy as np
from training.dataset import load_dataset


def census(path):
    data = load_dataset(Path(path))
    features = data['features']
    targets = data['probabilities']
    streets = features[:, 104:108].argmax(1)
    calls = features[:, 110] * 400
    odds = features[:, 252]
    events = features[:, 118:250].reshape(len(features), 12, 11)
    opponent_raises = (events[:, :, 1] * events[:, :, 9]).sum(1)

    def summarize(selection):
        count = int(selection.sum())
        selected = targets[selection]
        mass = selected.mean(0) if count else np.zeros(7)
        return {'count': count, 'legal_fold_count': int(data['masks'][selection, 0].sum()),
            'call_positive': int(((calls > 0) & selection).sum()),
            'pot_odds_bins': {
                'zero': int(((odds == 0) & selection).sum()),
                'positive_below_quarter': int(((odds > 0) & (odds < .25) & selection).sum()),
                'quarter_below_third': int(((odds >= .25) & (odds < 1 / 3 - 1e-6) & selection).sum()),
                'third_below_40pct': int(((odds >= 1 / 3 - 1e-6) & (odds < .4) & selection).sum()),
                'at_least_40pct': int(((odds >= .4) & selection).sum())},
            'call_size_bins_bb': {
                'zero': int(((calls == 0) & selection).sum()),
                'positive_at_most_one': int(((calls > 0) & (calls <= 4) & selection).sum()),
                'one_to_five': int(((calls > 4) & (calls <= 20) & selection).sum()),
                'five_below_25': int(((calls > 20) & (calls < 100) & selection).sum()),
                'at_least_25': int(((calls >= 100) & selection).sum())},
            'multiple_opponent_raises': int(((opponent_raises >= 2) & selection).sum()),
            'target_argmax_counts': np.bincount(selected.argmax(1), minlength=7).tolist() if count else [0] * 7,
            'mean_target_mass': mass.tolist(),
            'target_mass_groups': {'fold': float(mass[0]), 'check_call': float(mass[1]),
                'raise': float(mass[2:].sum())}}
    return {'path': str(path), 'metadata': data['metadata'],
        'overall': summarize(np.ones(len(features), dtype=bool)),
        'by_street': {str(street): summarize(streets == street) for street in range(4)},
        'pressure_subsets': {
            'postflop': summarize(streets > 0),
            'facing_at_least_25bb': summarize(calls >= 100),
            'pot_odds_at_least_40pct': summarize(odds >= .4)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='runs/heuristic-train.npz')
    parser.add_argument('--validation', default='runs/heuristic-validation.npz')
    parser.add_argument('--output', default='experiments/heuristic-coverage.json')
    args = parser.parse_args()
    report = {'premise': 'Imitation state sampling sufficiently covers fold and raise decisions before PPO.',
        'train': census(args.train), 'validation': census(args.validation),
        'interpretation': 'Street balance does not imply pressure balance. Pressure targets contain folds, '
            'but passive trajectories produce few large-bet or repeated-raise decisions.',
        'recommended_next_step': 'Test pressure-stratified legal trajectories with the unchanged labeler; '
            'retain independent natural-frequency tournament evaluation.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key]['overall'] for key in ('train', 'validation')}, indent=2))

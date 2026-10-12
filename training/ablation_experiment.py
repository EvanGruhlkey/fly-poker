import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_graph
from .dataset import load_dataset
from .features import FEATURE_COUNT, OBSERVATION_VERSION
from .model import BrainPolicy

MODES = ('real', 'shuffle', 'zero', 'mlp')


class SmallMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.trunk = nn.Sequential(nn.Linear(FEATURE_COUNT, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, 7)
        self.value = nn.Linear(hidden, 1)

    def forward(self, features, masks):
        encoded = self.trunk(features)
        return self.policy(encoded).masked_fill(~masks, -torch.inf), torch.tanh(self.value(encoded)).squeeze(-1)


def build_control(graph, mode):
    if mode not in MODES:
        raise ValueError('unknown control mode')
    return SmallMLP() if mode == 'mlp' else BrainPolicy(graph)


def control_loss(model, mode, batch):
    features, masks, probabilities, values = batch
    logits, prediction = model(features, masks) if mode == 'mlp' else model(features, masks, ablation=mode)
    logp = logits.log_softmax(-1).masked_fill(~masks, 0)
    return -(probabilities * logp).sum(-1).mean() + .5 * (prediction - values).square().mean()


def train_controls(graph, dataset, output, *, steps=64, seconds_per_mode=120,
                   seed=17, device='cpu', validation_dataset=None):
    if (not isinstance(steps, int) or not 1 <= steps <= 100000
            or not np.isfinite(seconds_per_mode) or not 0 <= seconds_per_mode <= 120
            or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise ValueError('invalid bounded control configuration')
    torch.set_num_threads(4)
    data = load_dataset(dataset, teacher='heuristic', split='train')
    rng = np.random.default_rng(seed)
    if validation_dataset is None:
        hands = np.unique(data['hand_seeds'])
        if len(hands) < 2:
            raise ValueError('at least two distinct hands required for heldout validation')
        rng.shuffle(hands)
        heldout = hands[:max(1, len(hands) // 5)]
        validation_indices = np.flatnonzero(np.isin(data['hand_seeds'], heldout))
        train_indices = np.flatnonzero(~np.isin(data['hand_seeds'], heldout))
        validation = {key: data[key][validation_indices] for key in ('features', 'masks', 'probabilities', 'values', 'hand_seeds')}
        validation_checksum = data['metadata']['checksum']
    else:
        validation = load_dataset(validation_dataset, teacher='heuristic', split='validation')
        if set(validation['hand_seeds']) & set(data['hand_seeds']):
            raise ValueError('training and validation hands overlap')
        train_indices = np.arange(len(data['values']))
        validation_checksum = validation['metadata']['checksum']
    indices = rng.choice(train_indices, size=(steps, 16), replace=True)
    digest = hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'batch-indices.npy', indices)
    config = {'steps': steps, 'seconds_per_mode': seconds_per_mode, 'seed': seed,
        'experiment_regime': 'wall-budget control' if steps == 100000 else 'matched-data step pilot',
        'training_budget_fraction': .9, 'batch': 16, 'optimizer': 'Adam', 'learning_rate': 3e-4,
        'initialization': 'from scratch', 'observation_version': OBSERVATION_VERSION,
        'graph_sha256': graph.sha256, 'dataset_checksum': data['metadata']['checksum'],
        'validation_checksum': validation_checksum, 'batch_indices_sha256': digest}
    report = {'config': config, 'modes': {},
        'resource_comparison': 'same step limit, wall limit and data batches; FLOPs differ',
        'shuffle_interpretation': 'fixed permutation of recurrent inputs; not an independent biological null',
        'train_hand_seeds': sorted(set(data['hand_seeds'][train_indices].tolist())),
        'validation_hand_seeds': sorted(set(validation['hand_seeds'].tolist()))}
    keys = ('features', 'masks', 'probabilities', 'values')
    def tensors(source, rows):
        return [torch.as_tensor(source[key][rows], device=device) for key in keys]
    for mode in MODES:
        torch.manual_seed(seed)
        started = time.monotonic()
        model = build_control(graph, mode).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        losses = []
        for rows in indices:
            if time.monotonic() - started >= seconds_per_mode * .9:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = control_loss(model, mode, tensors(data, rows))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite control loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            losses.append(float(loss.detach()))
        elapsed = time.monotonic() - started
        model.eval()
        validation_loss = None
        validation_rows = 0
        total = 0.
        with torch.no_grad():
            for offset in range(0, len(validation['values']), 16):
                if time.monotonic() - started >= seconds_per_mode:
                    break
                rows = np.arange(offset, min(offset + 16, len(validation['values'])))
                total += float(control_loss(model, mode, tensors(validation, rows))) * len(rows)
                validation_rows += len(rows)
        if validation_rows:
            validation_loss = total / validation_rows
        path = output / f'{mode}.pt'
        entry = {'actual_steps': len(losses), 'training_seconds': elapsed,
            'total_compute_seconds': time.monotonic() - started,
            'budget_exhausted': len(losses) < steps,
            'training_wall_limit_seconds': seconds_per_mode * .9,
            'stop_reason': 'step cap' if len(losses) == steps else 'training wall budget',
            'validation_loss': validation_loss, 'validation_rows': validation_rows,
            'validation_complete': validation_rows == len(validation['values']),
            'losses': losses, 'batch_indices_sha256': digest, 'checkpoint': str(path.resolve()),
            'parameter_count': sum(p.numel() for p in model.parameters())}
        torch.save({'format': 'fly-poker-architecture-control-v1', 'config': {**config, 'mode': mode},
                    'model_state': model.state_dict(), 'result': entry}, path)
        report['modes'][mode] = entry
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain matched-data architecture controls from scratch.')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--validation-dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seconds-per-mode', type=float, default=120)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cpu')
    args = vars(parser.parse_args())
    args['graph'] = load_graph(args['graph'])
    print(json.dumps(train_controls(**args), indent=2))

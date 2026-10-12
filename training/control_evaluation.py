import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ablation_experiment import MODES, build_control
from .batched_evaluation import atomic_report, evaluate_models, load_model
from .budget import Budget
from .checkpoint import FIXED_BUFFERS
from .data import load_graph
from .features import OBSERVATION_VERSION
from .tournament import TournamentConfig


class ControlPolicy(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unknown control mode')
        self.model = model
        self.mode = mode

    def forward(self, features, mask, ablation='real'):
        if self.mode == 'mlp':
            return self.model(features, mask)
        return self.model(features, mask, ablation=self.mode)


def load_control(path, graph, device):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload['config']
    if (payload.get('format') != 'fly-poker-architecture-control-v1'
            or config.get('graph_sha256') != graph.sha256
            or config.get('observation_version') != OBSERVATION_VERSION
            or not config.get('dataset_checksum') or config.get('mode') not in MODES):
        raise ValueError('incompatible architecture control')
    mode = config['mode']
    model = build_control(graph, mode).to(device)
    if mode != 'mlp':
        current = model.state_dict()
        if any(not torch.equal(payload['model_state'][key], current[key]) for key in FIXED_BUFFERS):
            raise ValueError('control modifies fixed connectome buffers')
    model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('nonfinite architecture control')
    wrapped = ControlPolicy(model, mode).eval()
    return wrapped, {'path':str(path.resolve()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'config':config, 'training_result':payload.get('result', {})}


def evaluate_controls(graph_path, control_dir, starting_path, output,
                      seconds_per_mode=120, pairs=32, device='cuda', seed=9418, commit=lambda: None):
    if (not np.isfinite(seconds_per_mode) or not 0 <= seconds_per_mode <= 300
            or not isinstance(pairs, int) or not 1 <= pairs <= 10000):
        raise ValueError('control evaluation exceeds bounded limits')
    config = TournamentConfig(pairs=pairs, seed=seed, domain='validation')
    graph = load_graph(graph_path)
    starting, starting_info = load_model(starting_path, graph, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'modes':{}, 'complete':False, 'source_sha256':source_sha,
        'interpretation':'Limited paired validation only; no Holdem strength or biological necessity claim.'}
    identity = None
    for mode in MODES:
        budget = Budget(seconds_per_mode)
        candidate, control_info = load_control(Path(control_dir) / f'{mode}.pt', graph, device)
        if control_info['config']['mode'] != mode:
            raise ValueError('control filename and trained ablation disagree')
        provenance = tuple(control_info['config'].get(key) for key in
            ('dataset_checksum','validation_checksum','batch_indices_sha256','seed','steps',
             'seconds_per_mode','batch','optimizer','learning_rate','initialization'))
        if identity is not None and provenance != identity:
            raise ValueError('architecture controls were trained on unmatched resources or data')
        identity = provenance
        metadata = {'artifacts':{'control':control_info,'starting':starting_info},
            'source_sha256':source_sha, 'elapsed_seconds':0., 'seconds_per_mode':seconds_per_mode,
            'interpretation':report['interpretation']}
        path = output / f'{mode}.json'
        def checkpoint(current):
            atomic_report(path, {**current, **metadata, 'elapsed_seconds':budget.elapsed})
            commit()
        result = evaluate_models(candidate, starting, budget, config, batch_pairs=8,
            opponents={'equity':None,'random':None,'v2':starting}, samples=32, checkpoint=checkpoint)
        result.update({**metadata,'elapsed_seconds':budget.elapsed})
        atomic_report(path,result)
        report['modes'][mode] = {'complete':result['complete'], 'elapsed_seconds':result['elapsed_seconds'],
            'report':str(path.resolve()), 'control_sha256':control_info['sha256']}
        report['complete'] = len(report['modes']) == len(MODES) and all(row['complete'] for row in report['modes'].values())
        atomic_report(output / 'report.json', report)
        commit()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score retrained architecture controls on heldout paired poker hands.')
    parser.add_argument('--graph-path',type=Path,required=True)
    parser.add_argument('--control-dir',type=Path,required=True)
    parser.add_argument('--starting-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds-per-mode',type=float,default=120)
    parser.add_argument('--pairs',type=int,default=32)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=9418)
    evaluate_controls(**vars(parser.parse_args()))

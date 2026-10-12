import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .budget import Budget
from .checkpoint import FIXED_BUFFERS
from .data import load_graph
from .environment import ACTION_VERSION
from .features import FEATURE_COUNT, OBSERVATION_VERSION
from .model import BrainPolicy, ModelConfig
from .tournament import TournamentConfig, baseline_policy, promotion, torch_policy, tournament


def load_policy(path, graph, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload['metadata']
    config = ModelConfig(**metadata['model'])
    versions = {118: 'cards-public-v1', 250: 'cards-history-v2', FEATURE_COUNT: OBSERVATION_VERSION}
    if (payload['format'] not in ('fly-poker-checkpoint-v2', 'fly-poker-checkpoint-v3', 'fly-poker-checkpoint-v4')
            or metadata['graph_sha256'] != graph.sha256
            or metadata['action_version'] != ACTION_VERSION
            or metadata['observation_version'] != versions.get(config.observation_features)):
        raise ValueError('incompatible evaluation checkpoint')
    model = BrainPolicy(graph, config).to(device)
    state = payload['model_state']
    if any(not torch.equal(state[key], model.state_dict()[key]) for key in FIXED_BUFFERS):
        raise ValueError('checkpoint modifies fixed connectome buffers')
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('checkpoint contains nonfinite parameters')
    return torch_policy(model), {'path': str(path.resolve()),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description='Evaluate paired held-out poker deals on the real connectome')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--starting', type=Path, required=True)
    parser.add_argument('--previous', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=9182)
    parser.add_argument('--domain', choices=('validation', 'test'), default='validation')
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--equity-samples', type=int, default=32)
    parser.add_argument('--match-hands', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    graph = load_graph(args.graph)
    candidate, candidate_info = load_policy(args.candidate, graph, args.device)
    starting, starting_info = load_policy(args.starting, graph, args.device)
    budget = Budget(args.seconds)
    opponents = {name: baseline_policy(name, samples=args.equity_samples, budget=budget)
                 for name in ('random', 'passive', 'aggressive', 'equity', 'teacher')}
    checkpoints = [starting_info]
    if starting_info['metadata']['model']['observation_features'] == 118:
        opponents['v2'] = starting
    else:
        opponents['starting'] = starting
    for index, path in enumerate(args.previous):
        policy, info = load_policy(path, graph, args.device)
        name = 'v2' if info['metadata']['model']['observation_features'] == 118 else f'previous_{index}'
        if name in opponents:
            raise ValueError(f'duplicate checkpoint opponent {name}')
        opponents[name] = policy
        checkpoints.append(info)
    report = tournament(candidate, starting, budget,
                        TournamentConfig(pairs=args.pairs, seed=args.seed, domain=args.domain,
                                         match_hands=args.match_hands), opponents=opponents)
    report['artifacts'] = {'candidate': candidate_info, 'starting': starting_info,
                           'opponent_checkpoints': checkpoints, 'graph_sha256': graph.sha256}
    report['equity_samples'] = args.equity_samples
    report['elapsed_seconds'] = budget.elapsed
    report['promotion'] = promotion(report, parity=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'complete': report['complete'],
                      'promotion': report['promotion']}, indent=2))


if __name__ == '__main__':
    main()

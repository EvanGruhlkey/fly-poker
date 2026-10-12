import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .budget import Budget, BudgetExceeded
from .checkpoint import FIXED_BUFFERS, migrate_v2, migrate_v3
from .data import load_graph
from .environment import ACTION_VERSION, Holdem
from .features import OBSERVATION_VERSION
from .model import BrainPolicy
from .rollout import infer
from .tournament import TournamentConfig, baseline_policy, promotion, summarize, _probabilities


def load_model(path, graph, device):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    model = BrainPolicy(graph).to(device)
    if payload['format'] == 'fly-poker-checkpoint-v2':
        migrate_v2(path, model)
    elif payload['format'] == 'fly-poker-checkpoint-v3':
        migrate_v3(path, model)
    else:
        metadata = payload['metadata']
        if (payload['format'] != 'fly-poker-checkpoint-v4'
                or metadata['graph_sha256'] != graph.sha256
                or metadata['model'] != model.metadata()['model']
                or metadata['observation_version'] != OBSERVATION_VERSION
                or metadata['action_version'] != ACTION_VERSION):
            raise ValueError('incompatible evaluation checkpoint')
        current = model.state_dict()
        if any(not torch.equal(payload['model_state'][key], current[key]) for key in FIXED_BUFFERS):
            raise ValueError('checkpoint modifies fixed connectome buffers')
        model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('nonfinite evaluation checkpoint')
    model.eval()
    return model, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'metadata': payload['metadata']}


def collect(model, opponent, kind, seeds, stacks, budget, samples):
    deals = [seed for seed in seeds for _ in (0, 1)]
    seats = [seat for _ in seeds for seat in (0, 1)]
    envs = [Holdem(seed, dealer=0, stacks=stacks if seat == 0 else stacks[::-1])
            for seed, seat in zip(deals, seats, strict=True)]
    learner_rngs = [np.random.default_rng(abs(seed) * 2 + seat)
                   for seed, seat in zip(deals, seats, strict=True)]
    opponent_rngs = [np.random.default_rng(abs(seed) * 2 + 1 - seat)
                    for seed, seat in zip(deals, seats, strict=True)]
    counts = [{} for _ in envs]
    last = [None for _ in envs]
    baseline = baseline_policy(kind, samples=samples, budget=budget) if opponent is None else None
    turns = 0
    while any(not env.terminal for env in envs):
        budget.check()
        turns += 1
        if turns > 400:
            raise RuntimeError('evaluation hand exceeded finite betting bound')
        indices = [i for i, env in enumerate(envs) if not env.terminal and env.actor == seats[i]]
        if indices:
            observations = [envs[i].observe() for i in indices]
            actions = infer(model, observations, generators=[learner_rngs[i] for i in indices])
            for i, obs, action in zip(indices, observations, actions, strict=True):
                key = f'street_{obs.street}_action_{action}'
                counts[i][key] = counts[i].get(key, 0) + 1
                last[i] = action
                envs[i].step(action)
        indices = [i for i, env in enumerate(envs) if not env.terminal and env.actor != seats[i]]
        if indices:
            observations = [envs[i].observe() for i in indices]
            if opponent is not None:
                actions = infer(opponent, observations, generators=[opponent_rngs[i] for i in indices])
            else:
                actions = [int(opponent_rngs[i].choice(7, p=_probabilities(baseline, obs)))
                           for i, obs in zip(indices, observations, strict=True)]
            for i, action in zip(indices, actions, strict=True):
                last[i] = action
                envs[i].step(action)
    return [(env.reward(seat), counts[i], last[i] != 0)
            for i, (env, seat) in enumerate(zip(envs, seats, strict=True))]


def statistics(seat_returns, counts, showdowns, wins, requested):
    pairs = [(a + b) / 2 for a, b in zip(seat_returns['0'], seat_returns['1'], strict=True)]
    hands = len(pairs) * 2
    totals = {str(action): sum(value for key, value in counts.items() if key.endswith(f'_action_{action}'))
              for action in range(7)}
    decisions = sum(totals.values())
    return {**summarize(pairs), 'pair_returns_bb': pairs, 'hands': hands,
            'requested_pairs': requested, 'complete': len(pairs) == requested,
            'seat_returns_bb': seat_returns, 'showdowns': showdowns, 'showdown_wins': wins,
            'seats': {seat: summarize(values) for seat, values in seat_returns.items()},
            'action_counts': counts,
            'action_frequencies': {key: value / decisions if decisions else None for key, value in totals.items()},
            'aggression_frequency': sum(totals[str(i)] for i in range(2, 7)) / decisions if decisions else None,
            'showdown_frequency': showdowns / hands if hands else None,
            'showdown_win_frequency': wins / showdowns if showdowns else None}


def entry(outcomes, requested, previous=None):
    seat_returns = {str(seat): [row[0] for row in outcomes[seat::2]] for seat in (0, 1)}
    counts = {}
    for _, actions, _ in outcomes:
        for action, count in actions.items():
            counts[action] = counts.get(action, 0) + count
    showdowns = sum(row[2] for row in outcomes)
    wins = sum(row[2] and row[0] > 0 for row in outcomes)
    if previous is not None:
        for seat in seat_returns:
            seat_returns[seat] = previous['seat_returns_bb'][seat] + seat_returns[seat]
        for action, count in previous['action_counts'].items():
            counts[action] = counts.get(action, 0) + count
        showdowns += previous['showdowns']
        wins += previous['showdown_wins']
    return statistics(seat_returns, counts, showdowns, wins, requested)


def validate_resume(saved, expected):
    for key in ('config', 'artifacts', 'baseline_samples', 'inference_batch_pairs'):
        if json.dumps(saved.get(key), sort_keys=True) != json.dumps(expected.get(key), sort_keys=True):
            raise ValueError(f'evaluation resume {key} mismatch')


def gain_entry(profile, kind, requested):
    gain = [a - b for a, b in zip(profile[f'candidate_vs_{kind}']['pair_returns_bb'],
                                 profile[f'starting_vs_{kind}']['pair_returns_bb'], strict=True)]
    return {**summarize(gain), 'pair_returns_bb': gain, 'complete': len(gain) == requested}


def evaluate_models(candidate, starting, budget, config, *, batch_pairs=8, opponents=None, samples=32,
                    prior=None, checkpoint=None):
    if not isinstance(batch_pairs, int) or not 1 <= batch_pairs <= 64 or not 1 <= samples <= 256:
        raise ValueError('invalid evaluation batch or sample count')
    if config.match_hands != 1:
        raise ValueError('batched evaluation supports one hand per paired deal')
    opponents = opponents if opponents is not None else {
        **dict.fromkeys(('random', 'passive', 'aggressive', 'equity', 'teacher')), 'starting': starting}
    if not opponents:
        raise ValueError('opponent panel must be nonempty')
    deal_seeds = config.deal_seeds
    report = {'config': asdict(config), 'deal_seeds': deal_seeds, 'profiles': {},
              'complete': False, 'budget_exhausted': False, 'policy_mode': 'sampled probabilities',
              'sample_unit': 'seat-swapped hand pair', 'inference_batch_pairs': batch_pairs,
              'baseline_samples': samples, 'sampling_rng': 'abs(deal_seed) * 2 + physical_seat',
              'resume_schema': 'paired-batched-evaluation-v1'}
    if prior is not None:
        validate_resume(prior, {**report, 'artifacts': prior.get('artifacts')})
        if prior.get('resume_schema') != report['resume_schema'] or prior.get('deal_seeds') != deal_seeds:
            raise ValueError('evaluation resume schema or deal manifest mismatch')
        report['profiles'] = deepcopy(prior['profiles'])
    expected_profiles = {f'{stacks[0]}_{stacks[1]}' for stacks in config.stack_profiles}
    if prior is not None and set(report['profiles']) != expected_profiles:
        raise ValueError('evaluation resume stack profiles mismatch')
    for stacks in config.stack_profiles:
        profile = report['profiles'].setdefault(f'{stacks[0]}_{stacks[1]}', {})
        expected_keys = {f'{label}_vs_{kind}' for label in ('candidate', 'starting', 'paired_gain') for kind in opponents}
        if prior is not None and set(profile) != expected_keys:
            raise ValueError('evaluation resume opponent panel mismatch')
        for kind in opponents:
            for label in ('candidate', 'starting'):
                key = f'{label}_vs_{kind}'
                if key not in profile:
                    profile[key] = entry([], config.pairs)
                row = profile[key]
                count = len(row['pair_returns_bb'])
                if (count > config.pairs or row.get('hands') != count * 2
                        or row.get('paired_samples') != count or count != len(row.get('seat_returns_bb', {}).get('0', []))
                        or count != len(row.get('seat_returns_bb', {}).get('1', []))):
                    raise ValueError('evaluation resume paired sample counts mismatch')
                if row['pair_returns_bb'] != [(a + b) / 2 for a, b in zip(row['seat_returns_bb']['0'], row['seat_returns_bb']['1'], strict=True)]:
                    raise ValueError('evaluation resume paired rewards mismatch')
            if profile[f'candidate_vs_{kind}']['paired_samples'] != profile[f'starting_vs_{kind}']['paired_samples']:
                raise ValueError('evaluation resume candidate/starting counts mismatch')
            for label in ('candidate', 'starting'):
                key = f'{label}_vs_{kind}'
                profile[key] = entry([], config.pairs, profile[key])
            profile[f'paired_gain_vs_{kind}'] = gain_entry(profile, kind, config.pairs)
    def update_status():
        report['complete'] = all(row['complete'] for profile in report['profiles'].values() for row in profile.values())
    try:
        for stacks in config.stack_profiles:
            profile = report['profiles'][f'{stacks[0]}_{stacks[1]}']
            for kind, opponent in opponents.items():
                completed = profile[f'candidate_vs_{kind}']['paired_samples']
                for offset in range(completed, config.pairs, batch_pairs):
                    budget.check()
                    seeds = deal_seeds[offset:offset + batch_pairs]
                    pair = {}
                    for label, model in (('candidate', candidate), ('starting', starting)):
                        pair[label] = collect(model, opponent, kind, seeds, stacks, budget, samples)
                    for label in pair:
                        key = f'{label}_vs_{kind}'
                        profile[key] = entry(pair[label], config.pairs, profile[key])
                    profile[f'paired_gain_vs_{kind}'] = gain_entry(profile, kind, config.pairs)
                    update_status()
                    if checkpoint:
                        checkpoint(report)
                if completed < config.pairs and profile[f'candidate_vs_{kind}']['complete']:
                    print(json.dumps({'phase': 'evaluation_panel', 'stack': list(stacks),
                        'opponent': kind, 'pairs': config.pairs,
                        'candidate_bb_per_100': profile[f'candidate_vs_{kind}']['bb_per_100'],
                        'gain_bb_per_100': profile[f'paired_gain_vs_{kind}']['bb_per_100'],
                        'elapsed_seconds': budget.elapsed}), flush=True)
    except BudgetExceeded:
        report['budget_exhausted'] = True
    update_status()
    return report


def atomic_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_checkpoints(graph_path, candidate_path, starting_path, output, budget_seconds=600,
                         pairs=1000, seed=9182, device='cuda', previous_paths=(), domain='validation',

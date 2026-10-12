import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .budget import Budget
from .environment import ACTION_VERSION, Holdem
from .features import FEATURE_COUNT, OBSERVATION_VERSION, encode
from .teacher import action_targets, equity, heuristic_targets, net_targets, sample_showdowns
from . import teacher as teacher_module

ARRAYS = ('features', 'masks', 'probabilities', 'values', 'hand_seeds')
SCHEMA = 'fly-poker-dataset-v2'
TEACHER_VERSIONS = {'equity': 'range-equity-net-v1', 'rollout': 'range-response-net-v2',
                    'heuristic': 'equity-heuristic-net-v1'}


def checksum(arrays, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key in ARRAYS:
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_dataset(path, *, seed, count, samples, budget, teacher='rollout', split='train', sampling='natural'):
    if (not isinstance(seed, int) or not 0 <= seed < 2**64
            or not isinstance(count, int) or not 1 <= count <= 100000
            or not isinstance(samples, int) or not 1 <= samples <= 256
            or teacher not in TEACHER_VERSIONS or split not in ('train', 'validation')
            or sampling not in ('natural', 'balanced-pressure')):
        raise ValueError('invalid dataset configuration')
    rng = np.random.default_rng(seed)
    features, masks, probabilities, values, seeds = [], [], [], [], []
    hand_index = 0
    while len(features) < count:
        budget.check()
        domain = 3 if split == 'train' else 4
        deal = domain * 2**192 + seed * 2**96 + hand_index
        stacks = (400, 400)
        if sampling == 'balanced-pressure' and rng.random() < .25:
            profiles = ((80, 720), (720, 80), (200, 600), (600, 200))
            stacks = profiles[int(rng.integers(len(profiles)))]
        env = Holdem(deal, dealer=hand_index % 2, stacks=stacks)
        hand_index += 1
        hand = []
        while not env.terminal:
            budget.check()
            obs = env.observe()
            hand.append(obs)
            if sampling == 'natural':
                action = 1 if rng.random() < .8 else int(rng.choice(np.flatnonzero(obs.mask)))
            else:
                behavior = int(rng.integers(3))
                if behavior == 0:
                    action = 1
                elif behavior == 1:
                    action = int(rng.choice(np.flatnonzero(obs.mask)))
                else:
                    raises = [index for index in (2, 3, 4) if obs.mask[index]]
                    action = int(rng.choice(raises)) if raises else 1
            env.step(action)
        if sampling == 'natural':
            matching = [obs for obs in hand if obs.street == len(features) % 4]
        else:
            wanted_street, wanted_pressure = divmod(len(features) % 12, 3)
            def pressure(observation):
                odds = observation.to_call / max(1, observation.pot + observation.to_call)
                return 0 if odds == 0 else 1 if odds <= .25 else 2
            matching = [obs for obs in hand
                if obs.street == wanted_street and pressure(obs) == wanted_pressure]
        if not matching:
            continue
        obs = matching[int(rng.integers(len(matching)))]
        if teacher == 'rollout':
            targets, value = net_targets(obs, sample_showdowns(obs, rng, samples, budget))
        elif teacher == 'heuristic':
            targets, value = heuristic_targets(obs, equity(obs, rng, samples, budget))
        else:
            targets, value = action_targets(obs, equity(obs, rng, samples, budget))
            value -= (obs.starting_stack - obs.stack) / 800
        features.append(encode(obs))
        masks.append(obs.mask)
        probabilities.append(targets)
        values.append(value)
        seeds.append(str(deal))
    arrays = {'features': np.asarray(features, dtype=np.float32),
              'masks': np.asarray(masks, dtype=bool),
              'probabilities': np.asarray(probabilities, dtype=np.float32),
              'values': np.asarray(values, dtype=np.float32), 'hand_seeds': np.asarray(seeds)}
    metadata = {'schema': SCHEMA, 'observation_version': OBSERVATION_VERSION,
                'action_version': ACTION_VERSION, 'teacher': teacher,
                'teacher_version': TEACHER_VERSIONS[teacher], 'seed': seed,
                'teacher_sha256': hashlib.sha256(Path(teacher_module.__file__).read_bytes()).hexdigest(),
                'count': count, 'samples': samples, 'split': split,
                'value_unit': 'net whole-hand chips / 800',
                'deal_domain': 'domain * 2**192 + seed * 2**96 + hand_index; train=3, validation=4'}
    metadata['sampling'] = sampling
    metadata['stack_curriculum'] = ('100 BB only' if sampling == 'natural' else
        '75% equal; 25% uniform 20/180 BB, 180/20 BB, 50/150 BB, 150/50 BB')
    metadata['checksum'] = checksum(arrays, metadata)
    budget.check()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays, metadata=json.dumps(metadata, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
            budget.check()
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**arrays, 'metadata': metadata}


def load_dataset(path, **expected):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in ARRAYS}
        metadata = json.loads(str(source['metadata']))
    if (metadata.get('schema') != SCHEMA or metadata.get('observation_version') != OBSERVATION_VERSION
            or metadata.get('action_version') != ACTION_VERSION
            or metadata.get('teacher_version') != TEACHER_VERSIONS.get(metadata.get('teacher'))):
        raise ValueError('incompatible dataset schema or versions')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('dataset configuration mismatch')
    manifest = {key: value for key, value in metadata.items() if key != 'checksum'}
    if checksum(arrays, manifest) != metadata.get('checksum'):
        raise ValueError('dataset checksum mismatch')
    count = metadata['count']
    if (arrays['features'].shape != (count, FEATURE_COUNT) or arrays['masks'].shape != (count, 7)
            or arrays['probabilities'].shape != (count, 7) or arrays['values'].shape != (count,)
            or arrays['hand_seeds'].shape != (count,) or arrays['masks'].dtype != bool
            or any(not np.isfinite(arrays[key]).all() for key in ARRAYS[:4])
            or (arrays['probabilities'] < 0).any()
            or not np.allclose(arrays['probabilities'].sum(axis=1), 1)
            or arrays['probabilities'][~arrays['masks']].any()):
        raise ValueError('invalid dataset arrays')
    return {**arrays, 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description='Build reusable poker teacher targets without graph inference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--count', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--sampling', choices=('natural', 'balanced-pressure'), default='natural')
    parser.add_argument('--teacher', choices=tuple(TEACHER_VERSIONS), default='equity')
    args = parser.parse_args()
    data = build_dataset(args.output, seed=args.seed, count=args.count, samples=args.samples,
                         budget=Budget(args.seconds), split=args.split, teacher=args.teacher, sampling=args.sampling)
    print(json.dumps(data['metadata'], indent=2))


if __name__ == '__main__':
    main()

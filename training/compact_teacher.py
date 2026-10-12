import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from .budget import Budget, BudgetExceeded
from .dataset import load_dataset
from .features import FEATURE_COUNT, encode
from .rollout import rollout, train_batch, TRAIN_SEED_LIMIT


class CompactTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(FEATURE_COUNT, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Linear(64, 7)
        self.value = nn.Linear(64, 1)

    def forward(self, features, mask, ablation='real'):
        if features.shape != (len(features), FEATURE_COUNT) or mask.shape != (len(features), 7) or not mask.any(-1).all():
            raise ValueError('invalid compact teacher observations')
        hidden = self.hidden(features)
        return self.policy(hidden).masked_fill(~mask, -torch.inf), self.value(hidden).tanh().squeeze(-1)


def loadcompact(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['format'] != 'compact-training-teacher-v1' or payload['features'] != FEATURE_COUNT:
        raise ValueError('incompatible compact teacher checkpoint')
    model = CompactTeacher().to(device)
    model.load_state_dict(payload['model_state'])
    model.eval().requires_grad_(False)

    def policy(obs):
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(encode(obs), device=device).unsqueeze(0),
                torch.tensor([obs.mask], dtype=torch.bool, device=device))
            return logits.softmax(-1)[0].cpu().numpy()
    return policy


def _loss(model, dataset, indices, device):
    features, masks, probabilities, values = [torch.as_tensor(dataset[key][indices], device=device)
        for key in ('features', 'masks', 'probabilities', 'values')]
    logits, predicted = model(features, masks)
    return -(probabilities * logits.log_softmax(-1).masked_fill(~masks, 0)).sum(-1).mean() + .5 * (predicted - values).square().mean()


def run(dataset, output, *, seconds=60, updates=256, warm_steps=256, batch=128,
        seed=23, validation_dataset=None, device='cpu', resume=False, learning_rate=3e-4):
    if (not 0 < seconds <= 300 or not 0 <= updates <= 10000 or not 0 <= warm_steps <= 10000
            or batch not in (64, 128) or not 0 <= seed < TRAIN_SEED_LIMIT
            or not 0 < learning_rate <= .01):
        raise ValueError('compact teacher run exceeds supported bounds')
    budget = Budget(seconds)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = load_dataset(Path(dataset), teacher='heuristic', split='train')
    validation = load_dataset(Path(validation_dataset), teacher='heuristic', split='validation') if validation_dataset else None
    if validation and set(data['hand_seeds']) & set(validation['hand_seeds']):
        raise ValueError('training and validation hands overlap')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {'seed': seed, 'batch': batch, 'learning_rate': learning_rate,
        'dataset_checksum': data['metadata']['checksum'],
        'validation_checksum': validation['metadata']['checksum'] if validation else None}
    model = CompactTeacher().to(device)
    frozen = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    progress = {'warm_steps': 0, 'updates': 0, 'hands': 0}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'latest.pt'
    if resume:
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved['format'] != 'compact-training-teacher-v1' or saved['config'] != config or saved['source_sha256'] != source_hash:
            raise ValueError('compact teacher resume configuration mismatch')
        model.load_state_dict(saved['model_state'])
        frozen.load_state_dict(saved['frozen_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        progress = saved['progress']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    elif path.exists():
        raise ValueError('compact teacher output already exists; use resume')

    def checkpoint():
        payload = {'format': 'compact-training-teacher-v1', 'features': FEATURE_COUNT,
            'source_sha256': source_hash, 'model_state': model.state_dict(),
            'frozen_state': frozen.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
            'progress': progress.copy(), 'config': config}
        with tempfile.NamedTemporaryFile(dir=output, suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(payload, stream)
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

    losses = []
    exhausted = False
    checkpoint()
    try:
        while progress['warm_steps'] < warm_steps:
            budget.check()
            indices = rng.integers(len(data['values']), size=batch)
            loss = _loss(model, data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            progress['warm_steps'] += 1
            losses.append(float(loss.detach()))
        if progress['updates'] == 0:
            frozen.load_state_dict(model.state_dict())
        while progress['updates'] < updates:
            budget.check()
            old_state = deepcopy(model.state_dict())
            old_optimizer = deepcopy(optimizer.state_dict())
            old_numpy = deepcopy(rng.bit_generator.state)
            old_torch = torch.get_rng_state()
            old_cuda = torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []
            try:
                seeds = rng.integers(TRAIN_SEED_LIMIT, size=batch).tolist()
                seats = rng.integers(2, size=batch).tolist()
                opponents = rng.choice(['random', 'check_call', 'equity', 'frozen'], size=batch).tolist()
                stacks = [(400, 400) if rng.random() < .75 else
                    (buyin := int(rng.integers(80, 721)), 800 - buyin) for _ in seeds]
                episodes = rollout(model, frozen, rng, budget, seeds=seeds, learner_seats=seats,
                    opponents=opponents, dealers=rng.integers(2, size=batch).tolist(), initial_stacks=stacks)
                stats = train_batch(model, optimizer, episodes, budget, epochs=2, minibatch_size=128)
            except BudgetExceeded:
                model.load_state_dict(old_state)
                optimizer.load_state_dict(old_optimizer)
                rng.bit_generator.state = old_numpy
                torch.set_rng_state(old_torch)
                if old_cuda:
                    torch.cuda.set_rng_state_all(old_cuda)
                raise
            progress['updates'] += 1
            progress['hands'] += batch
            losses.append(stats['loss'])
            if progress['updates'] % 16 == 0:
                frozen.load_state_dict(model.state_dict())
            checkpoint()
    except BudgetExceeded:
        exhausted = True
    checkpoint()
    validation_loss = None
    if validation:
        with torch.no_grad():
            validation_loss = float(_loss(model, validation, slice(None), device))
    report = {'progress': progress, 'config': config, 'source_sha256': source_hash,
        'budget_exhausted': exhausted, 'elapsed_seconds': budget.elapsed,
        'validation_loss': validation_loss, 'recent_losses': losses[-8:],
        'value_units': 'net terminal BB / 200', 'checkpoint': str(path),
        'role': 'compact experimental training teacher; no production fly export',
        'architecture': f'{FEATURE_COUNT}->64->64->masked7/tanhvalue'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='runs/heuristic-train.npz')
    parser.add_argument('--validation-dataset', default='runs/heuristic-validation.npz')
    parser.add_argument('--output', default='runs/compact-teacher')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--warm-steps', type=int, default=256)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--resume', action='store_true')
    arguments = vars(parser.parse_args())
    print(json.dumps(run(**arguments), indent=2))

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import torch
from .budget import Budget, BudgetExceeded
from .checkpoint import load_checkpoint, save_checkpoint
from .data import Graph
from .evaluation import evaluate
from .model import BrainPolicy
from .rollout import rollout, tensor_batch, train_batch, TRAIN_SEED_LIMIT
from .teacher import equity, heuristic, sample_states


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    warm_steps: int = 64
    updates: int = 32
    batch: int = 8
    eval_pairs: int = 15
    resume: bool = False

    def validate(self):
        if not 0 <= self.seed < TRAIN_SEED_LIMIT or not 1 <= self.batch <= 16:
            raise ValueError('seed or batch outside supported bounds')
        if not 0 <= self.warm_steps <= 256 or not 0 <= self.updates <= 256 or not 0 <= self.eval_pairs <= 50:
            raise ValueError('training or evaluation limit outside supported bounds')


def copy_parameters(source, target):
    with torch.no_grad():
        for destination, original in zip(target.parameters(), source.parameters(), strict=True):
            destination.copy_(original)


def train(graph: Graph, directory: Path, config: TrainConfig, budget: Budget,
          *, device='cpu', commit=lambda: None) -> dict:
    config.validate()
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = BrainPolicy(graph).to(device)
    untrained = BrainPolicy(graph).to(device)
    frozen = BrainPolicy(graph).to(device)
    copy_parameters(model, untrained)
    copy_parameters(model, frozen)
    initial_sensory = model.sensory.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    progress = {'warm_steps': 0, 'updates': 0, 'hands': 0, 'decisions': 0}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'latest.pt'
    if config.resume:
        restored = load_checkpoint(path, model, optimizer, rng)
        if restored['config']['seed'] != config.seed or restored['config']['batch'] != config.batch:
            raise ValueError('resume seed and batch must match the saved run')
        progress = restored['progress']
        copy_parameters(model, frozen)
    else:
        save_checkpoint(directory / 'initial.pt', model, optimizer, rng, progress.copy(), asdict(config))
    def checkpoint():
        save_checkpoint(path, model, optimizer, rng, progress.copy(), asdict(config))
        commit()
    checkpoint()
    losses = []
    exhausted = False
    try:
        while progress['warm_steps'] < config.warm_steps:
            budget.check()
            observations = sample_states(rng, config.batch, budget)
            equities = [equity(obs, rng, budget=budget) for obs in observations]
            targets = torch.tensor([heuristic(obs, probability) for obs, probability in zip(observations, equities, strict=True)], device=device)
            values = torch.tensor([2 * probability - 1 for probability in equities], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = model(*tensor_batch(observations, device))
            loss = torch.nn.functional.cross_entropy(logits, targets) + .5 * (prediction - values).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            progress['warm_steps'] += 1
            losses.append(float(loss.detach()))
            if progress['warm_steps'] % 8 == 0:
                checkpoint()
                print(json.dumps({'phase': 'warm', **progress, 'elapsed_seconds': budget.elapsed}), flush=True)
        while progress['updates'] < config.updates:
            budget.check()
            copy_parameters(model, frozen)
            seeds = rng.integers(0, TRAIN_SEED_LIMIT, config.batch).tolist()
            seats = rng.integers(0, 2, config.batch).tolist()
            opponents = rng.choice(['random', 'check_call', 'equity', 'frozen'], config.batch).tolist()
            episodes = rollout(model, frozen, rng, budget, seeds=seeds, learner_seats=seats, opponents=opponents)
            result = train_batch(model, optimizer, episodes, budget)
            progress['updates'] += 1
            progress['hands'] += len(episodes)
            progress['decisions'] += result['decisions']
            losses.append(result['loss'])
            if progress['updates'] % 4 == 0:
                checkpoint()
                print(json.dumps({'phase': 'actor_critic', **progress, 'elapsed_seconds': budget.elapsed}), flush=True)
    except BudgetExceeded:
        exhausted = True
    checkpoint()
    evaluations = evaluate(model, untrained, budget, seed=config.seed, pairs=config.eval_pairs) if config.eval_pairs else {}
    exhausted |= any(item['budget_exhausted'] for item in evaluations.values())
    report = {'metadata': model.metadata(), 'config': asdict(config), 'progress': progress,
        'budget_exhausted': exhausted, 'elapsed_seconds': budget.elapsed,
        'sensory_weights_changed': not torch.equal(initial_sensory, model.sensory.weight.detach()),
        'recent_losses': losses[-8:], 'evaluation': evaluations, 'checkpoint': str(path),
        'training_seed_domain': [0, TRAIN_SEED_LIMIT - 1], 'evaluation_seed_domain': [TRAIN_SEED_LIMIT, None],
        'interpretation': 'Fixed signed connectome rate approximation with trained adapters; pilot results do not establish poker strength.'}
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    commit()
    return report


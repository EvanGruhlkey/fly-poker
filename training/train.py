from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
import numpy as np
import torch
from .budget import Budget, BudgetExceeded
from .checkpoint import load_checkpoint, save_checkpoint, migrate_v2
from .data import Graph
from .evaluation import evaluate
from .model import BrainPolicy
from .rollout import rollout, tensor_batch, train_batch, TRAIN_SEED_LIMIT
from .teacher import equity, action_targets, sample_states, initial_stacks


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    warm_steps: int = 64
    updates: int = 32
    batch: int = 8
    eval_pairs: int = 15
    resume: bool = False
    init_from_v2: str | None = None

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
    migration = migrate_v2(Path(config.init_from_v2), model) if config.init_from_v2 and not config.resume else None
    untrained = BrainPolicy(graph).to(device)
    frozen = BrainPolicy(graph).to(device)
    copy_parameters(model, untrained)
    copy_parameters(model, frozen)
    initial_edges = model.syn_gain.detach().clone()
    initial_leaks = model.leak_logit.detach().clone()
    initial_sensory = model.sensory.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    progress = {'warm_steps': 0, 'updates': 0, 'hands': 0, 'decisions': 0, 'self_play_hands': 0}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'latest.pt'
    if config.resume:
        restored = load_checkpoint(path, model, optimizer, rng)
        if restored['config']['seed'] != config.seed or restored['config']['batch'] != config.batch:
            raise ValueError('resume seed and batch must match the saved run')
        progress = restored['progress']
        if config.init_from_v2:
            migrate_v2(Path(config.init_from_v2), untrained)
        copy_parameters(model, frozen)
    else:
        save_checkpoint(directory / 'initial.pt', model, optimizer, rng, progress.copy(), asdict(config))
    def checkpoint():
        save_checkpoint(path, model, optimizer, rng, progress.copy(), asdict(config))
        commit()
    checkpoint()
    remaining = max(0.0, budget.deadline - time.monotonic())
    evaluation_reserve = min(180.0, remaining * .25) if config.eval_pairs else 0.0
    training_budget = Budget(remaining - evaluation_reserve)
    losses = []
    exhausted = False
    try:
        while progress['warm_steps'] < config.warm_steps:
            training_budget.check()
            observations = sample_states(rng, config.batch, training_budget)
            equities = [equity(obs, rng, budget=training_budget) for obs in observations]
            teacher_targets = [action_targets(obs, probability) for obs, probability in zip(observations, equities, strict=True)]
            targets = torch.tensor(np.stack([target for target, _ in teacher_targets]), device=device)
            values = torch.tensor([value for _, value in teacher_targets], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = model(*tensor_batch(observations, device))
            loss = -(targets * logits.log_softmax(-1).masked_fill(~torch.tensor([obs.mask for obs in observations], device=device), 0)).sum(-1).mean() + .5 * (prediction - values).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            progress['warm_steps'] += 1
            losses.append(float(loss.detach()))
            if progress['warm_steps'] % 8 == 0:
                checkpoint()
                print(json.dumps({'phase': 'warm', **progress, 'elapsed_seconds': budget.elapsed}), flush=True)
        while progress['updates'] < config.updates:
            training_budget.check()
            copy_parameters(model, frozen)
            seeds = rng.integers(0, TRAIN_SEED_LIMIT, config.batch).tolist()
            seats = rng.integers(0, 2, config.batch).tolist()
            opponents = rng.choice(['random', 'check_call', 'equity', 'frozen', 'frozen', 'frozen'], config.batch).tolist()
            stacks = [initial_stacks(rng, variable=progress['updates'] >= 128) for _ in seeds]
            episodes = rollout(model, frozen, rng, training_budget, seeds=seeds, learner_seats=seats, opponents=opponents, initial_stacks=stacks)
            result = train_batch(model, optimizer, episodes, training_budget)
            progress['updates'] += 1
            progress['hands'] += len(episodes)
            progress['self_play_hands'] += opponents.count('frozen')
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
        'evaluation_reserved_seconds': evaluation_reserve, 'reward_discount': 1.0,
        'frozen_opponent': 'snapshot refreshed before every actor-critic update',
        'device': str(device), 'gpu_name': torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        'edge_magnitudes_changed': not torch.equal(initial_edges, model.syn_gain.detach()),
        'neuron_leaks_changed': not torch.equal(initial_leaks, model.leak_logit.detach()),
        'fixed_signs_preserved': torch.equal(model.edge_weights().sign().to(torch.int8), model.signs),
        'parameters_finite': all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
        'sensory_weights_changed': not torch.equal(initial_sensory, model.sensory.weight.detach()),
        'recent_losses': losses[-8:], 'evaluation': evaluations, 'checkpoint': str(path),
        'migration': migration, 'teacher': {'samples': 64, 'targets': 'soft incremental showdown value; conservative always-call response', 'range': 'importance weighted preflop strength prior conditioned on public aggression'},
        'starting_policy': 'migrated-v2' if config.init_from_v2 else 'random-initialization',
        'curriculum': '100 BB first 128 updates, then 75% 100 BB and 25% unequal stacks',
        'training_seed_domain': [0, TRAIN_SEED_LIMIT - 1], 'evaluation_seed_domain': [TRAIN_SEED_LIMIT, None],
        'interpretation': 'Fixed connectome topology/signs with learned edge magnitudes, neuron leaks, and adapters; pilot results do not establish poker strength.'}
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    commit()
    return report


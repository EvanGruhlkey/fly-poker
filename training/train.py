from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
import numpy as np
import torch
from .budget import Budget, BudgetExceeded
from .checkpoint import load_checkpoint, save_checkpoint, migrate_v2, migrate_v3
from .data import Graph
from .evaluation import evaluate
from .model import BrainPolicy
from .rollout import rollout, tensor_batch, train_batch, TRAIN_SEED_LIMIT
from .teacher import equity, heuristic_targets, sample_states, initial_stacks
from .league import OpponentLeague
from .dataset import load_dataset


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    warm_steps: int = 64
    updates: int = 32
    batch: int = 8
    eval_pairs: int = 15
    resume: bool = False
    init_from_v2: str | None = None
    init_from_v3: str | None = None
    adapter_steps: int = 64
    league_interval: int = 32
    league_capacity: int = 3
    ppo_epochs: int = 4
    dataset: str | None = None
    validation_dataset: str | None = None
    learning_rate: float = 3e-4
    teacher_report: str | None = None

    def validate(self):
        if not 0 <= self.seed < TRAIN_SEED_LIMIT or not 1 <= self.batch <= 16:
            raise ValueError('seed or batch outside supported bounds')
        if not 0 <= self.warm_steps <= 100000 or not 0 <= self.updates <= 100000 or not 0 <= self.eval_pairs <= 1000:
            raise ValueError('training or evaluation limit outside supported bounds')
        if not 0 <= self.adapter_steps <= 100000 or not 1 <= self.ppo_epochs <= 8 or not 2 <= self.league_interval <= 1000 or not 1 <= self.league_capacity <= 4:
            raise ValueError('stage or league controls outside supported bounds')
        if self.init_from_v2 and self.init_from_v3:
            raise ValueError('choose one migration source')
        if not 0 < self.learning_rate <= .01:
            raise ValueError('learning rate outside supported bounds')


def copy_parameters(source, target):
    with torch.no_grad():
        for destination, original in zip(target.parameters(), source.parameters(), strict=True):
            destination.copy_(original)


def supervision_data(config):
    dataset = load_dataset(Path(config.dataset), split='train') if config.dataset else None
    validation = load_dataset(Path(config.validation_dataset), split='validation') if config.validation_dataset else None
    if dataset and dataset['metadata']['teacher'] == 'rollout':
        import hashlib
        from . import teacher
        from .teacher_benchmark import validation_gate
        report = json.loads(Path(config.teacher_report).read_text()) if config.teacher_report else {}
        current_teacher = hashlib.sha256(Path(teacher.__file__).read_bytes()).hexdigest()
        if (not validation_gate(report)['accepted'] or report.get('teacher_sha256') != current_teacher
                or dataset['metadata'].get('teacher_sha256') != current_teacher
                or report.get('samples') != dataset['metadata']['samples']):
            raise ValueError('rollout teacher requires a passing benchmark for the current teacher code and samples')
    return dataset, validation


def train(graph: Graph, directory: Path, config: TrainConfig, budget: Budget,
          *, device='cpu', commit=lambda: None) -> dict:
    config.validate()
    dataset, validation = supervision_data(config)
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = BrainPolicy(graph).to(device)
    migration = migrate_v2(Path(config.init_from_v2), model) if config.init_from_v2 and not config.resume else None
    if config.init_from_v3 and not config.resume:
        migration = migrate_v3(Path(config.init_from_v3), model)
    untrained = BrainPolicy(graph).to(device)
    frozen = BrainPolicy(graph).to(device)
    copy_parameters(model, untrained)
    copy_parameters(model, frozen)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    progress = {'warm_steps': 0, 'updates': 0, 'hands': 0, 'decisions': 0, 'self_play_hands': 0}
    fingerprints = {'dataset_checksum': dataset['metadata']['checksum'] if dataset else None,
                    'validation_checksum': validation['metadata']['checksum'] if validation else None}
    progress.update(fingerprints)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'latest.pt'
    if not config.resume and (path.exists() or (directory / 'league').exists()):
        raise ValueError('fresh experiments require a new run directory')
    if config.resume:
        restored = load_checkpoint(path, model, optimizer, rng)
        mutable = ('resume', 'warm_steps', 'updates', 'eval_pairs')
        if any(restored['config'].get(key) != value for key, value in asdict(config).items() if key not in mutable):
            raise ValueError('resume experiment configuration must match the saved run')
        progress = restored['progress']
        if any(progress.get(key) != value for key, value in fingerprints.items()):
            raise ValueError('resume dataset checksum mismatch')
        if config.init_from_v2:
            migration = migrate_v2(Path(config.init_from_v2), untrained)
        elif config.init_from_v3:
            migration = migrate_v3(Path(config.init_from_v3), untrained)
        else:
            initial = torch.load(directory / 'initial.pt', map_location=device, weights_only=True)
            untrained.load_state_dict(initial['model_state'])
        copy_parameters(model, frozen)
    else:
        save_checkpoint(directory / 'initial.pt', model, optimizer, rng, progress.copy(), asdict(config))
    persistence = {'save_seconds': 0., 'commit_seconds': 0., 'commits': 0}
    last_commit = -float('inf')
    def checkpoint(force=False):
        nonlocal last_commit
        started = time.monotonic()
        save_checkpoint(path, model, optimizer, rng, progress.copy(), asdict(config))
        persistence['save_seconds'] += time.monotonic() - started
        if force or time.monotonic() - last_commit >= 60:
            started = time.monotonic()
            commit()
            persistence['commit_seconds'] += time.monotonic() - started
            persistence['commits'] += 1
            last_commit = time.monotonic()
    checkpoint(force=True)
    initial = torch.load(directory / 'initial.pt', map_location=device, weights_only=True)
    initial_edges = initial['model_state']['syn_gain']
    initial_leaks = initial['model_state']['leak_logit']
    initial_sensory = initial['model_state']['sensory.weight']
    league = OpponentLeague(model, directory / 'league', config.league_interval,
                            config.league_capacity, update=progress['updates'])
    anchors = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
               if name in ('syn_gain', 'bias', 'leak_logit')}
    if config.resume:
        anchors = {name: initial['model_state'][name] for name in anchors}
    def configure_stage():
        adapter = progress['warm_steps'] + progress['updates'] < config.adapter_steps
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not adapter or name not in anchors)
        return 'adapter' if adapter else 'connectome'
    def supervised_loss(features, masks, targets, values):
        logits, prediction = model(features, masks)
        cross_entropy = -(targets * logits.log_softmax(-1).masked_fill(~masks, 0)).sum(-1).mean()
        return cross_entropy + .5 * (prediction - values).square().mean()
    remaining = max(0.0, budget.deadline - time.monotonic())
    evaluation_reserve = min(180.0, remaining * .25) if config.eval_pairs else 0.0
    training_budget = Budget(remaining - evaluation_reserve)
    losses = []
    exhausted = False
    try:
        while progress['warm_steps'] < config.warm_steps:
            training_budget.check()
            step_started = time.monotonic()
            stage = configure_stage()
            if dataset:
                indices = rng.integers(len(dataset['values']), size=config.batch)
                features, masks, targets, values = [torch.as_tensor(dataset[key][indices], device=device)
                    for key in ('features', 'masks', 'probabilities', 'values')]
            else:
                observations = sample_states(rng, config.batch, training_budget)
                equities = [equity(obs, rng, budget=training_budget) for obs in observations]
                teacher_targets = [heuristic_targets(obs, probability) for obs, probability in zip(observations, equities, strict=True)]
                targets = torch.tensor(np.stack([target for target, _ in teacher_targets]), device=device)
                values = torch.tensor([value for _, value in teacher_targets], device=device)
                features, masks = tensor_batch(observations, device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(features, masks, targets, values)
            if stage == 'connectome':
                loss = loss + .01 * sum((parameter - anchors[name]).square().mean()
                    for name, parameter in model.named_parameters() if name in anchors)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in anchors and parameter.requires_grad:
                        parameter.copy_(torch.maximum(torch.minimum(parameter, anchors[name] + .1), anchors[name] - .1))
            progress['warm_steps'] += 1
            if progress['warm_steps'] == config.adapter_steps and not progress['updates']:
                save_checkpoint(directory / 'adapter.pt', model, optimizer, rng,
                    progress.copy(), asdict(config))
            losses.append(float(loss.detach()))
            with (directory / 'learning.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'phase': 'imitation', 'stage': stage,
                    'progress': progress.copy(), 'loss': losses[-1],
                    'target_action_mass': targets.detach().mean(0).cpu().tolist(),
                    'step_seconds': time.monotonic() - step_started,
                    'elapsed_seconds': budget.elapsed}) + '\n')
            if progress['warm_steps'] % 8 == 0:
                checkpoint()
                print(json.dumps({'phase': 'warm', **progress, 'elapsed_seconds': budget.elapsed}), flush=True)
        if config.warm_steps and not progress['updates']:
            save_checkpoint(directory / 'imitation.pt', model, optimizer, rng,
                progress.copy(), asdict(config))
        while progress['updates'] < config.updates:
            training_budget.check()
            configure_stage()
            copy_parameters(model, frozen)
            seeds = rng.integers(0, TRAIN_SEED_LIMIT, config.batch).tolist()
            seats = rng.integers(0, 2, config.batch).tolist()
            policies = league.policies()
            opponents = rng.choice(['random', 'check_call', 'equity', 'teacher', 'frozen', *policies], config.batch).tolist()
            stacks = [initial_stacks(rng, variable=progress['updates'] >= 128) for _ in seeds]
            rollout_started = time.monotonic()
            episodes = rollout(model, frozen, rng, training_budget, seeds=seeds, learner_seats=seats,
                opponents=opponents, initial_stacks=stacks, dealers=rng.integers(0, 2, config.batch).tolist(),
                opponent_policies=policies)
            rollout_seconds = time.monotonic() - rollout_started
            gradient_started = time.monotonic()
            result = train_batch(model, optimizer, episodes, training_budget, epochs=config.ppo_epochs,
                                 anchors=anchors)
            gradient_seconds = time.monotonic() - gradient_started
            progress['updates'] += 1
            progress['hands'] += len(episodes)
            progress['self_play_hands'] += sum(name == 'frozen' or name in policies for name in opponents)
            progress['decisions'] += result['decisions']
            losses.append(result['loss'])
            league.refresh(model, progress['updates'])
            with (directory / 'learning.jsonl').open('a', encoding='utf-8') as stream:
                counts = np.bincount([a for episode in episodes for a in episode.actions], minlength=7).tolist()
                stream.write(json.dumps({'phase': 'ppo', 'progress': progress.copy(), 'stats': result,
                    'rollout_seconds': rollout_seconds, 'gradient_seconds': gradient_seconds,
                    'action_counts': counts,
                    'mean_reward_bb': float(np.mean([episode.reward_bb for episode in episodes])),
                    'elapsed_seconds': budget.elapsed}) + '\n')
            if progress['updates'] % 4 == 0:
                checkpoint()
                print(json.dumps({'phase': 'actor_critic', **progress, 'elapsed_seconds': budget.elapsed}), flush=True)
    except BudgetExceeded:
        exhausted = True
        restored = load_checkpoint(path, model, optimizer, rng)
        progress = restored['progress']
        league = OpponentLeague(model, directory / 'league', config.league_interval,
                                config.league_capacity, update=progress['updates'])
    checkpoint(force=True)
    validation_loss = None
    if validation and not exhausted:
        try:
            validation_losses = []
            with torch.no_grad():
                for start in range(0, len(validation['values']), config.batch):
                    budget.check()
                    validation_losses.append(float(supervised_loss(*[
                        torch.as_tensor(validation[key][start:start + config.batch], device=device)
                        for key in ('features', 'masks', 'probabilities', 'values')])))
            validation_loss = float(np.mean(validation_losses))
        except BudgetExceeded:
            exhausted = True
    evaluations = evaluate(model, untrained, budget, seed=config.seed, pairs=config.eval_pairs) if config.eval_pairs else {}
    exhausted |= any(item['budget_exhausted'] for item in evaluations.values())
    report = {'metadata': model.metadata(), 'config': asdict(config), 'progress': progress,
        'budget_exhausted': exhausted, 'elapsed_seconds': budget.elapsed,
        'checkpoint_persistence': {**persistence, 'remote_commit_interval_seconds': 60},
        'evaluation_reserved_seconds': evaluation_reserve, 'reward_discount': 1.0,
        'frozen_opponent': league.metadata(), 'value_units': 'net terminal BB / 200',
        'validation_loss': validation_loss, 'dataset': dataset['metadata'] if dataset else None,
        'device': str(device), 'gpu_name': torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        'edge_magnitudes_changed': not torch.equal(initial_edges, model.syn_gain.detach()),
        'neuron_leaks_changed': not torch.equal(initial_leaks, model.leak_logit.detach()),
        'fixed_signs_preserved': torch.equal(model.edge_weights().sign().to(torch.int8), model.signs),
        'parameters_finite': all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
        'sensory_weights_changed': not torch.equal(initial_sensory, model.sensory.weight.detach()),
        'recent_losses': losses[-8:], 'evaluation': evaluations, 'checkpoint': str(path),
        'migration': migration, 'teacher': {'samples': 64 if dataset is None else dataset['metadata']['samples'],
            'targets': 'soft action probabilities with expected net whole-hand return',
            'kind': 'heuristic' if dataset is None else dataset['metadata']['teacher'],
            'range': 'heuristic public-aggression prior'},
        'starting_policy': 'migrated-v2' if config.init_from_v2 else 'migrated-v3' if config.init_from_v3 else 'random-initialization',
        'curriculum': '100 BB first 128 updates, then 75% 100 BB and 25% unequal stacks',
        'training_seed_domain': [0, TRAIN_SEED_LIMIT - 1], 'evaluation_seed_domain': [TRAIN_SEED_LIMIT, None],
        'interpretation': 'Fixed connectome topology/signs with learned edge magnitudes, neuron leaks, and adapters; pilot results do not establish poker strength.'}
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    commit()
    return report


if __name__ == '__main__':
    import argparse
    from .data import load_graph
    parser = argparse.ArgumentParser(description='Run a bounded staged fly poker experiment')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 900:
        parser.error('seconds must be finite and between 0 and 900')
    controls = json.loads(args.config.read_text(encoding='utf-8'))
    controls.pop('run_name', None)
    controls['resume'] = args.resume
    print(json.dumps(train(load_graph(args.graph), args.output, TrainConfig(**controls),
                           Budget(args.seconds), device=args.device), indent=2))


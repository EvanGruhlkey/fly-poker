import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .budget import Budget, BudgetExceeded
from .checkpoint import load_checkpoint, migrate_v2, migrate_v3, save_checkpoint
from .data import load_graph
from .environment import ACTION_NAMES, Holdem
from .model import BrainPolicy
from .rollout import tensor_batch
from .teacher import action_rewards, sample_showdowns, sample_states

FROZEN = ('syn_gain', 'leak_logit', 'bias')
CONNECTORS = ('sensory.weight', 'sensory.bias', 'motor_norm.weight', 'motor_norm.bias', 'policy.weight', 'policy.bias', 'value.weight', 'value.bias')


def trainable(model: BrainPolicy):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in FROZEN)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def optimizer_for(model: BrainPolicy, lr: float):
    return torch.optim.Adam(trainable(model), lr=lr)


def hole_policy(model: BrainPolicy, observation, hole: tuple[str, str]) -> tuple[list[float], int]:
    view = replace(observation, hole=hole)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, _ = model(*tensor_batch([view], device))
    probabilities = logits.softmax(-1).squeeze(0).cpu().numpy()
    return [round(float(probability), 4) for probability in probabilities], int(np.argmax(probabilities))


def separation(observation, aces: tuple[list[float], int], junk: tuple[list[float], int]) -> bool:
    aces_kind = observation.actions[aces[1]][0]
    junk_kind = observation.actions[junk[1]][0]
    return aces_kind == 'raise' and junk_kind in ('fold', 'check_call') and aces[1] != junk[1]


def describe(observation, policy: tuple[list[float], int]) -> dict:
    kind, amount = observation.actions[policy[1]]
    return {'probabilities': dict(zip(ACTION_NAMES, policy[0], strict=True)), 'action': kind, 'amount': amount}


def dopamine_loss(logits: torch.Tensor, prediction: torch.Tensor, rewards: torch.Tensor, actions: torch.Tensor, entropy: float = .1) -> torch.Tensor:
    distribution = torch.distributions.Categorical(logits=logits)
    reward = rewards.gather(1, actions[:, None]).squeeze(1)
    advantage = (reward - prediction).detach()
    policy = -(distribution.log_prob(actions) * advantage).mean()
    value = .5 * (prediction - reward).square().mean()
    return policy + value - entropy * distribution.entropy().mean()


def recurrence_effect(model: BrainPolicy, observation) -> float:
    device = next(model.parameters()).device
    features, mask = tensor_batch([observation], device)
    with torch.no_grad():
        live, _ = model(features, mask, ablation='real')
        dead, _ = model(features, mask, ablation='zero')
    finite = torch.isfinite(live) & torch.isfinite(dead)
    return float((live[finite] - dead[finite]).abs().mean())


def fit(model: BrainPolicy, optimizer, rng: np.random.Generator, budget: Budget, *, steps: int, batch: int, samples: int, entropy: float) -> list[float]:
    device = next(model.parameters()).device
    losses = []
    for _ in range(steps):
        budget.check()
        observations, rewards = [], []
        for observation in sample_states(rng, batch, budget):
            observations.append(observation)
            rewards.append(action_rewards(observation, sample_showdowns(observation, rng, samples, budget)))
        logits, prediction = model(*tensor_batch(observations, device))
        reward = torch.tensor(np.stack(rewards), device=device)
        draws = torch.distributions.Categorical(logits=logits).sample((4,))
        loss = dopamine_loss(logits.repeat(4, 1), prediction.repeat(4), reward.repeat(4, 1), draws.reshape(-1), entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        illegal = [name for name, parameter in model.named_parameters() if parameter.grad is not None and name not in CONNECTORS and name not in FROZEN]
        if illegal:
            raise RuntimeError(f'gradient outside the fly connectors: {illegal}')
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run(graph_path: Path, init_path: Path, output: Path, *, steps: int, batch: int, samples: int, wall_seconds: float, seed: int, lr: float, publish: Path | None, device: str = 'cpu', entropy: float = .1) -> dict:
    if publish is not None:
        raise ValueError('probe-only poker fit cannot publish a production model; held-out promotion is required')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = BrainPolicy(load_graph(graph_path)).to(device)
    optimizer = optimizer_for(model, lr)
    rng = np.random.default_rng(seed)
    header = torch.load(init_path, map_location=device, weights_only=True)
    if header['format'] == 'fly-poker-checkpoint-v2':
        migration = migrate_v2(init_path, model)
    elif header['format'] == 'fly-poker-checkpoint-v3':
        migration = migrate_v3(init_path, model)
    elif header['format'] == 'fly-poker-checkpoint-v4':
        restored = load_checkpoint(init_path, model, optimizer, rng)
        migration = {'resumed': str(init_path), 'progress': restored['progress']}
    else:
        raise ValueError('init checkpoint must be fly poker v2, v3, or v4')
    spot = Holdem(seed).observe()
    holes = {'aces': ('As', 'Ah'), 'kings': ('Ks', 'Kh'), 'junk': ('7d', '2c')}
    before = {name: hole_policy(model, spot, hole) for name, hole in holes.items()}
    budget = Budget(wall_seconds)
    completed = 0
    losses = []
    exhausted = False
    try:
        while completed < steps:
            losses.extend(fit(model, optimizer, rng, budget, steps=1, batch=batch, samples=samples, entropy=entropy))
            completed += 1
    except BudgetExceeded:
        exhausted = True
    after = {name: hole_policy(model, spot, hole) for name, hole in holes.items()}
    passed = separation(spot, after['aces'], after['junk'])
    output.mkdir(parents=True, exist_ok=True)
    progress = {'warm_steps': completed, 'updates': completed, 'hands': completed * batch, 'decisions': completed * batch}
    config = {'seed': seed, 'steps': steps, 'batch': batch, 'samples': samples, 'lr': lr, 'entropy': entropy, 'objective': 'fly-dopamine', 'frozen': list(FROZEN), 'connectors': list(CONNECTORS)}
    save_checkpoint(output / 'latest.pt', model, optimizer, rng, progress, config)
    report = {
        'objective': 'The fly samples its own action from motor neurons. Dopamine is the reward of that action minus the value prediction. Only the state connector and the reward connector learn.',
        'recurrence_effect': round(recurrence_effect(model, spot), 6),
        'migration': migration, 'steps': completed, 'budget_exhausted': exhausted, 'elapsed_seconds': round(budget.elapsed, 1),
        'passed': passed, 'published': bool(passed and publish is not None),
        'device': device, 'gpu_name': torch.cuda.get_device_name() if device.startswith('cuda') else None,
        'all_in_gap': round(describe(spot, after['aces'])['probabilities']['all_in'] - describe(spot, after['junk'])['probabilities']['all_in'], 4),
        'before': {name: describe(spot, policy) for name, policy in before.items()},
        'after': {name: describe(spot, policy) for name, policy in after.items()},
        'last_losses': losses[-8:], 'checkpoint': str(output / 'latest.pt'),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fit the fly to poker decision values with fold equity.')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--wall-seconds', type=float, default=900)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    print(json.dumps(run(args.graph, args.init, args.output, steps=args.steps, batch=args.batch, samples=args.samples,
        wall_seconds=args.wall_seconds, seed=args.seed, lr=args.lr, publish=args.publish), indent=2))

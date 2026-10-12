from dataclasses import dataclass
import numpy as np
import torch
from .budget import Budget
from .environment import Holdem, Observation
from .features import encode
from .model import BrainPolicy
from .teacher import baseline

TRAIN_SEED_LIMIT = 2**30
EVAL_SEED_BASE = 2**30


@dataclass(frozen=True)
class Episode:
    observations: tuple[Observation, ...]
    actions: tuple[int, ...]
    reward_bb: float
    seed: int
    learner_seat: int
    log_probs: tuple[float, ...] = ()
    values: tuple[float, ...] = ()


def tensor_batch(observations: list[Observation], device):
    features = torch.from_numpy(np.stack([encode(obs) for obs in observations])).to(device)
    mask = torch.tensor([obs.mask for obs in observations], dtype=torch.bool, device=device)
    return features, mask


def infer(model: BrainPolicy, observations: list[Observation], *, greedy=False, ablation='real', generators=None) -> list[int]:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, _ = model(*tensor_batch(observations, device), ablation=ablation)
    return _sample(logits, greedy=greedy, generators=generators)


def _sample(logits, *, greedy=False, generators=None) -> list[int]:
    if greedy:
        return logits.argmax(-1).cpu().tolist()
    if generators is None:
        return torch.distributions.Categorical(logits=logits).sample().cpu().tolist()
    probabilities = logits.softmax(-1).cpu().numpy()
    return [int(generator.choice(7, p=probability / probability.sum()))
        for generator, probability in zip(generators, probabilities, strict=True)]


def rollout(model: BrainPolicy, frozen: BrainPolicy, rng: np.random.Generator,
            budget: Budget, *, seeds: list[int], learner_seats: list[int],
            opponents: list[str], greedy=False, ablation='real', initial_stacks=None,
            dealers: list[int] | None = None,
            opponent_policies: dict[str, BrainPolicy] | None = None) -> list[Episode]:
    if not len(seeds) == len(learner_seats) == len(opponents):
        raise ValueError('rollout seat, seed, and opponent lengths differ')
    starting = initial_stacks or [(400, 400)] * len(seeds)
    dealer_seats = dealers if dealers is not None else [0] * len(seeds)
    policies = {'frozen': frozen, **(opponent_policies or {})}
    envs = [Holdem(seed, dealer=dealer, stacks=stacks)
        for seed, stacks, dealer in zip(seeds, starting, dealer_seats, strict=True)]
    observations: list[list[Observation]] = [[] for _ in seeds]
    actions: list[list[int]] = [[] for _ in seeds]
    action_rngs = [np.random.default_rng(seed * 2 + seat) for seed, seat in zip(seeds, learner_seats, strict=True)]
    opponent_rngs = [np.random.default_rng(seed * 2 + 1 - seat) for seed, seat in zip(seeds, learner_seats, strict=True)]
    log_probs: list[list[float]] = [[] for _ in seeds]
    values: list[list[float]] = [[] for _ in seeds]
    turns = 0
    while any(not env.terminal for env in envs):
        budget.check()
        turns += 1
        if turns > 400:
            raise RuntimeError('hand exceeded the finite betting-step bound')
        learner_indices = [i for i, env in enumerate(envs) if not env.terminal and env.actor == learner_seats[i]]
        if learner_indices:
            batch = [envs[i].observe() for i in learner_indices]
            with torch.no_grad():
                logits, predicted_values = model(*tensor_batch(batch, next(model.parameters()).device), ablation=ablation)
                choices = _sample(logits, greedy=greedy, generators=[action_rngs[i] for i in learner_indices])
                behavior_logs = torch.distributions.Categorical(logits=logits).log_prob(
                    torch.tensor(choices, device=logits.device)).cpu().tolist()
            for index, obs, action, log_prob, value in zip(learner_indices, batch, choices,
                    behavior_logs, predicted_values.cpu().tolist(), strict=True):
                log_probs[index].append(log_prob)
                values[index].append(value)
                observations[index].append(obs)
                actions[index].append(action)
                envs[index].step(action)
        frozen_indices: dict[str, list[int]] = {}
        for i, env in enumerate(envs):
            if env.terminal or env.actor == learner_seats[i]:
                continue
            if opponents[i] in policies:
                frozen_indices.setdefault(opponents[i], []).append(i)
            else:
                env.step(baseline(env.observe(), opponents[i], opponent_rngs[i], budget))
        for name, indices in frozen_indices.items():
            batch = [envs[i].observe() for i in indices]
            for index, action in zip(indices, infer(policies[name], batch, generators=[opponent_rngs[i] for i in indices]), strict=True):
                envs[index].step(action)
    return [Episode(tuple(obs), tuple(action), env.reward(seat), seed, seat, tuple(logs), tuple(predictions))
        for obs, action, env, seed, seat, logs, predictions in zip(observations, actions, envs, seeds,
            learner_seats, log_probs, values, strict=True)]


def normalized_reward(reward_bb: float) -> float:
    return reward_bb / 200


def train_batch(model: BrainPolicy, optimizer, episodes: list[Episode], budget: Budget,
                *, epochs: int = 4, minibatch_size: int = 16, clip_ratio: float = .2,
                anchors: dict | None = None, drift_limit: float = .1,
                entropy_coefficient: float = .01, target_kl: float | None = .03) -> dict:
    if epochs < 1 or minibatch_size < 1 or not 0 < clip_ratio < 1:
        raise ValueError('invalid PPO update configuration')
    if drift_limit <= 0 or not 0 <= entropy_coefficient <= .1 or (target_kl is not None and target_kl <= 0):
        raise ValueError('invalid PPO stability configuration')
    anchored = [(parameter, anchors[name].to(parameter.device))
        for name, parameter in model.named_parameters()
        if anchors and name in anchors and parameter.requires_grad]
    observations, actions, targets, old_logs, old_values = [], [], [], [], []
    device = next(model.parameters()).device
    for episode in episodes:
        count = len(episode.actions)
        if len(episode.observations) != count:
            raise ValueError('episode observations and actions differ')
        if not count:
            continue
        if episode.log_probs or episode.values:
            if len(episode.log_probs) != count or len(episode.values) != count:
                raise ValueError('episode behavior data and actions differ')
            logs, values = episode.log_probs, episode.values
        else:
            with torch.no_grad():
                logits, predictions = model(*tensor_batch(list(episode.observations), device))
                logs = torch.distributions.Categorical(logits=logits).log_prob(
                    torch.tensor(episode.actions, device=device)).cpu().tolist()
                values = predictions.cpu().tolist()
        observations.extend(episode.observations)
        actions.extend(episode.actions)
        targets.extend([normalized_reward(episode.reward_bb)] * count)
        old_logs.extend(logs)
        old_values.extend(values)
    if not observations:
        return {'loss': 0.0, 'decisions': 0, 'epochs': 0, 'updates': 0,
            'approx_kl': 0.0, 'clip_fraction': 0.0, 'value_loss': 0.0, 'advantage_mean': 0.0, 'early_stop': False}
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    behavior_log = torch.tensor(old_logs, dtype=torch.float32, device=device)
    advantage = target - torch.tensor(old_values, dtype=torch.float32, device=device)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    action = torch.tensor(actions, device=device)
    totals = dict(loss=0.0, approx_kl=0.0, clip_fraction=0.0, value_loss=0.0)
    updates = 0
    processed = 0
    early_stop = False
    completed_epochs = 0
    for epoch in range(epochs):
        completed_epochs = epoch + 1
        order = torch.randperm(len(observations)).tolist()
        for start in range(0, len(order), minibatch_size):
            budget.check()
            indices = order[start:start + minibatch_size]
            logits, value = model(*tensor_batch([observations[i] for i in indices], device))
            distribution = torch.distributions.Categorical(logits=logits)
            log_ratio = distribution.log_prob(action[indices]) - behavior_log[indices]
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.minimum(ratio * advantage[indices], clipped * advantage[indices]).mean()
            value_loss = (value - target[indices]).square().mean()
            loss = policy_loss + .5 * value_loss - entropy_coefficient * distribution.entropy().mean()
            if anchored:
                loss = loss + .01 * sum((parameter - anchor).square().mean()
                    for parameter, anchor in anchored)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for parameter, anchor in anchored:
                    parameter.copy_(torch.maximum(anchor - drift_limit,
                        torch.minimum(parameter, anchor + drift_limit)))
            weight = len(indices)
            totals['loss'] += float(loss.detach()) * weight
            totals['value_loss'] += float(value_loss.detach()) * weight
            approximate_kl = float((ratio - 1 - log_ratio).mean().detach())
            totals['approx_kl'] += approximate_kl * weight
            totals['clip_fraction'] += float(((ratio - 1).abs() > clip_ratio).float().mean()) * weight
            updates += 1
            processed += weight
            if target_kl is not None and approximate_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break
    totals = {name: value / processed for name, value in totals.items()}
    return {**totals, 'decisions': len(observations), 'epochs': completed_epochs, 'updates': updates,
        'early_stop': early_stop,
        'advantage_mean': float(advantage.mean())}

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


def tensor_batch(observations: list[Observation], device):
    features = torch.from_numpy(np.stack([encode(obs) for obs in observations])).to(device)
    mask = torch.tensor([obs.mask for obs in observations], dtype=torch.bool, device=device)
    return features, mask


def infer(model: BrainPolicy, observations: list[Observation], *, greedy=False, ablation='real') -> list[int]:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, _ = model(*tensor_batch(observations, device), ablation=ablation)
        actions = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
    return actions.cpu().tolist()


def rollout(model: BrainPolicy, frozen: BrainPolicy, rng: np.random.Generator,
            budget: Budget, *, seeds: list[int], learner_seats: list[int],
            opponents: list[str], greedy=False, ablation='real', initial_stacks=None) -> list[Episode]:
    if not len(seeds) == len(learner_seats) == len(opponents):
        raise ValueError('rollout seat, seed, and opponent lengths differ')
    starting = initial_stacks or [(400, 400)] * len(seeds)
    envs = [Holdem(seed, dealer=0, stacks=stacks) for seed, stacks in zip(seeds, starting, strict=True)]
    observations: list[list[Observation]] = [[] for _ in seeds]
    actions: list[list[int]] = [[] for _ in seeds]
    turns = 0
    while any(not env.terminal for env in envs):
        budget.check()
        turns += 1
        if turns > 400:
            raise RuntimeError('hand exceeded the finite betting-step bound')
        learner_indices = [i for i, env in enumerate(envs) if not env.terminal and env.actor == learner_seats[i]]
        if learner_indices:
            batch = [envs[i].observe() for i in learner_indices]
            choices = infer(model, batch, greedy=greedy, ablation=ablation)
            for index, obs, action in zip(learner_indices, batch, choices, strict=True):
                observations[index].append(obs)
                actions[index].append(action)
                envs[index].step(action)
        frozen_indices = []
        for i, env in enumerate(envs):
            if env.terminal or env.actor == learner_seats[i]:
                continue
            if opponents[i] == 'frozen':
                frozen_indices.append(i)
            else:
                env.step(baseline(env.observe(), opponents[i], rng, budget))
        if frozen_indices:
            batch = [envs[i].observe() for i in frozen_indices]
            for index, action in zip(frozen_indices, infer(frozen, batch), strict=True):
                envs[index].step(action)
    return [Episode(tuple(obs), tuple(action), env.reward(seat), seed, seat)
        for obs, action, env, seed, seat in zip(observations, actions, envs, seeds, learner_seats, strict=True)]


def train_batch(model: BrainPolicy, optimizer, episodes: list[Episode], budget: Budget) -> dict:
    observations = []
    actions = []
    targets = []
    for episode in episodes:
        count = len(episode.actions)
        observations.extend(episode.observations)
        actions.extend(episode.actions)
        targets.extend(episode.reward_bb / 100 for index in range(count))
    if not observations:
        return {'loss': 0.0, 'decisions': 0}
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for start in range(0, len(observations), 16):
        budget.check()
        stop = min(len(observations), start + 16)
        logits, value = model(*tensor_batch(observations[start:stop], device))
        distribution = torch.distributions.Categorical(logits=logits)
        target = torch.tensor(targets[start:stop], dtype=torch.float32, device=device)
        action = torch.tensor(actions[start:stop], device=device)
        advantage = target - value.detach()
        loss = (-(distribution.log_prob(action) * advantage).mean() + .5 * (value - target).square().mean() - .01 * distribution.entropy().mean())
        weighted = loss * (stop - start) / len(observations)
        weighted.backward()
        total_loss += float(weighted.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {'loss': total_loss, 'decisions': len(observations)}

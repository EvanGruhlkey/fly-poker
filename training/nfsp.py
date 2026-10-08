from copy import deepcopy
import numpy as np
import torch
from torch import nn


def masked_probabilities(logits, mask):
    if logits.shape != mask.shape or not mask.any(-1).all():
        raise ValueError('invalid NFSP legal mask')
    return logits.masked_fill(~mask, -torch.inf).softmax(-1)


class Memory:
    def __init__(self, capacity, *, reservoir=False):
        if capacity < 1:
            raise ValueError('memory capacity must be positive')
        self.capacity, self.reservoir = capacity, reservoir
        self.items, self.seen = [], 0

    def add(self, item, rng):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = int(rng.integers(self.seen)) if self.reservoir else (self.seen - 1) % self.capacity
            if index < self.capacity:
                self.items[index] = item

    def sample(self, count, rng):
        indices = rng.choice(len(self.items), min(count, len(self.items)), replace=False)
        return [self.items[int(index)] for index in indices]

    def state(self):
        return {'items': self.items, 'seen': self.seen, 'capacity': self.capacity, 'reservoir': self.reservoir}

    def restore(self, state):
        if state['capacity'] != self.capacity or state['reservoir'] != self.reservoir:
            raise ValueError('NFSP memory configuration mismatch')
        self.items, self.seen = state['items'], state['seen']


def network(features, actions, hidden):
    return nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, actions))


class NFSPAgent:
    def __init__(self, features, actions, *, hidden=64, capacity=50000, lr=.001):
        self.q = network(features, actions, hidden)
        self.target = deepcopy(self.q)
        self.average = network(features, actions, hidden)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.average_optimizer = torch.optim.Adam(self.average.parameters(), lr=lr)
        self.replay = Memory(capacity)
        self.supervised = Memory(capacity, reservoir=True)
        self.updates = 0

    def act(self, features, mask, best_response, epsilon, rng):
        features, mask = torch.as_tensor(features).float(), torch.as_tensor(mask).bool()
        with torch.no_grad():
            if best_response:
                legal = np.flatnonzero(mask.numpy())
                action = int(rng.choice(legal)) if rng.random() < epsilon else int(self.q(features).masked_fill(~mask, -torch.inf).argmax())
                self.supervised.add((features.clone(), mask.clone(), action), rng)
            else:
                probabilities = masked_probabilities(self.average(features), mask).numpy()
                action = int(rng.choice(len(mask), p=probabilities / probabilities.sum()))
        return action

    def transition(self, features, mask, action, reward, next_features, next_mask, terminal, rng):
        self.replay.add((torch.as_tensor(features).float().clone(), torch.as_tensor(mask).bool().clone(),
                         action, float(reward), torch.as_tensor(next_features).float().clone(),
                         torch.as_tensor(next_mask).bool().clone(), bool(terminal)), rng)

    def learn(self, rng, *, batch=64, target_every=200):
        losses = {}
        if len(self.replay.items) >= batch:
            rows = self.replay.sample(batch, rng)
            features = torch.stack([row[0] for row in rows])
            actions = torch.tensor([row[2] for row in rows])
            rewards = torch.tensor([row[3] for row in rows])
            future = torch.stack([row[4] for row in rows])
            legal = torch.stack([row[5] for row in rows])
            terminal = torch.tensor([row[6] for row in rows])
            with torch.no_grad():
                values = self.target(future).masked_fill(~legal, -torch.inf).max(-1).values
                target = rewards + torch.where(terminal, 0., values)
            prediction = self.q(features).gather(1, actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(prediction, target)
            self.q_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 5)
            self.q_optimizer.step()
            losses['q_loss'] = float(loss.detach())
            self.updates += 1
            if self.updates % target_every == 0:
                self.target.load_state_dict(self.q.state_dict())
        if len(self.supervised.items) >= batch:
            rows = self.supervised.sample(batch, rng)
            features = torch.stack([row[0] for row in rows])
            legal = torch.stack([row[1] for row in rows])
            actions = torch.tensor([row[2] for row in rows])
            logits = self.average(features).masked_fill(~legal, -torch.inf)
            loss = nn.functional.cross_entropy(logits, actions)
            self.average_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.average.parameters(), 5)
            self.average_optimizer.step()
            losses['average_loss'] = float(loss.detach())
        return losses

    def state(self):
        return {**{name: getattr(self, name).state_dict() for name in
                   ('q', 'target', 'average', 'q_optimizer', 'average_optimizer')},
                'replay': self.replay.state(), 'supervised': self.supervised.state(), 'updates': self.updates}

    def restore(self, state):
        for name in ('q', 'target', 'average', 'q_optimizer', 'average_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.replay.restore(state['replay'])
        self.supervised.restore(state['supervised'])
        self.updates = state['updates']

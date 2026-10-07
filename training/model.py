from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from .data import Graph
from .features import FEATURE_COUNT


@dataclass(frozen=True)
class ModelConfig:
    steps: int = 8
    gain: float = 4.0
    observation_features: int = FEATURE_COUNT
    action_count: int = 7
    approximation: str = 'signed-leaky-rate-reservoir-v1'


class BrainPolicy(nn.Module):
    def __init__(self, graph: Graph, config: ModelConfig = ModelConfig()):
        super().__init__()
        graph.validate()
        if config.steps != 8 or config.observation_features != FEATURE_COUNT or config.action_count != 7:
            raise ValueError('incompatible observation, action, or recurrence configuration')
        self.config = config
        self.graph_sha = graph.sha256
        rows = np.repeat(np.arange(graph.n), np.diff(graph.indptr))
        row_total = np.bincount(rows, weights=graph.counts, minlength=graph.n)
        weights = graph.signs * graph.counts / np.maximum(1, row_total[rows])
        adjacency = torch.sparse_csr_tensor(torch.from_numpy(graph.indptr.astype(np.int64)),
            torch.from_numpy(graph.indices.astype(np.int64)), torch.tensor(weights, dtype=torch.float32),
            size=(graph.n, graph.n))
        self.register_buffer('adjacency', adjacency)
        self.register_buffer('root_ids', torch.tensor(graph.root_ids, dtype=torch.int64))
        self.register_buffer('signs', torch.tensor(graph.signs, dtype=torch.int8))
        self.register_buffer('input_idx', torch.tensor(graph.inputs, dtype=torch.int64))
        self.register_buffer('output_idx', torch.tensor(graph.outputs, dtype=torch.int64))
        self.register_buffer('shuffle_idx', torch.randperm(graph.n, generator=torch.Generator().manual_seed(9182)), persistent=False)
        self.sensory = nn.Linear(FEATURE_COUNT, len(graph.inputs))
        self.bias = nn.Parameter(torch.zeros(graph.n))
        self.leak_logit = nn.Parameter(torch.tensor(.8))
        self.motor_norm = nn.LayerNorm(len(graph.outputs))
        self.policy = nn.Linear(len(graph.outputs), 7)
        self.value = nn.Linear(len(graph.outputs), 1)

    def propagate(self, rates: torch.Tensor, ablation: str = 'real') -> torch.Tensor:
        if ablation == 'zero':
            return torch.zeros_like(rates)
        if ablation == 'shuffle':
            rates = rates.index_select(1, self.shuffle_idx)
        elif ablation != 'real':
            raise ValueError('unknown graph ablation')
        return torch.sparse.mm(self.adjacency, rates.T).T

    def forward(self, features: torch.Tensor, mask: torch.Tensor, ablation: str = 'real'):
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT or mask.shape != (len(features), 7) or not mask.any(dim=1).all():
            raise ValueError('invalid structured observation or legal-action mask')
        rates = torch.zeros(len(features), self.adjacency.shape[0], device=features.device)
        sensory = rates.index_copy(1, self.input_idx, self.sensory(features))
        leak = .1 + .8 * torch.sigmoid(self.leak_logit)
        for _ in range(self.config.steps):
            activation = torch.sigmoid(self.config.gain * self.propagate(rates, ablation) + sensory + self.bias)
            rates = (1 - leak) * rates + leak * activation
        motor = self.motor_norm(rates.index_select(1, self.output_idx))
        return self.policy(motor).masked_fill(~mask, -torch.inf), torch.tanh(self.value(motor)).squeeze(-1)

    def metadata(self) -> dict:
        return {'model': asdict(self.config), 'graph_sha256': self.graph_sha,
            'neurons': self.adjacency.shape[0], 'connections': self.adjacency._nnz(),
            'inputs': len(self.input_idx), 'outputs': len(self.output_idx)}

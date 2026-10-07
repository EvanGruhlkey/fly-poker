from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from .data import Graph
from .features import FEATURE_COUNT
from .sparse import signed_mm


@dataclass(frozen=True)
class ModelConfig:
    steps: int = 8
    gain: float = 1.0
    saturation: float = 10.0
    observation_features: int = FEATURE_COUNT
    action_count: int = 7
    approximation: str = 'signed-learned-saturating-rate-v2'


class BrainPolicy(nn.Module):
    def __init__(self, graph: Graph, config: ModelConfig = ModelConfig()):
        super().__init__()
        graph.validate()
        if config.steps != 8 or config.observation_features != FEATURE_COUNT or config.action_count != 7:
            raise ValueError('incompatible observation, action, or recurrence configuration')
        self.config = config
        self.graph_sha = graph.sha256
        rows = np.repeat(np.arange(graph.n), np.diff(graph.indptr))
        permutation = np.argsort(graph.indices, kind='stable')
        transptr = np.concatenate(([0], np.cumsum(np.bincount(graph.indices, minlength=graph.n))))
        for name, array in (('csr_indptr', graph.indptr), ('csr_indices', graph.indices),
                            ('root_ids', graph.root_ids), ('input_idx', graph.inputs), ('output_idx', graph.outputs)):
            self.register_buffer(name, torch.tensor(array, dtype=torch.int64))
        self.register_buffer('signs', torch.tensor(graph.signs, dtype=torch.int8))
        for name, array in (('transpose_indptr', transptr), ('transpose_indices', rows[permutation]),
                            ('transpose_permutation', permutation)):
            self.register_buffer(name, torch.tensor(array, dtype=torch.int64), persistent=False)
        self.register_buffer('shuffle_idx', torch.randperm(graph.n, generator=torch.Generator().manual_seed(9182)), persistent=False)
        counts = torch.tensor(np.log1p(graph.counts), dtype=torch.float32)
        magnitude = (config.gain * counts / (counts.sum() / graph.n).clamp(min=1e-6)).clamp(min=1e-6)
        self.syn_gain = nn.Parameter(magnitude + torch.log(-torch.expm1(-magnitude)))
        self.sensory = nn.Linear(FEATURE_COUNT, len(graph.inputs))
        self.bias = nn.Parameter(torch.zeros(graph.n))
        self.leak_logit = nn.Parameter(torch.zeros(graph.n))
        self.motor_norm = nn.LayerNorm(len(graph.outputs))
        self.policy = nn.Linear(len(graph.outputs), 7)
        self.value = nn.Linear(len(graph.outputs), 1)

    def edge_weights(self):
        return self.signs * torch.nn.functional.softplus(self.syn_gain)

    @property
    def adjacency(self):
        n = len(self.bias)
        return torch.sparse_csr_tensor(self.csr_indptr, self.csr_indices, self.edge_weights(), size=(n, n))

    def propagate(self, rates: torch.Tensor, ablation: str = 'real', weights=None) -> torch.Tensor:
        if ablation == 'zero':
            return torch.zeros_like(rates)
        if ablation == 'shuffle':
            rates = rates.index_select(1, self.shuffle_idx)
        elif ablation != 'real':
            raise ValueError('unknown graph ablation')
        return signed_mm(self.edge_weights() if weights is None else weights, rates.T, self).T

    def rates(self, features: torch.Tensor, ablation: str = 'real'):
        rates = torch.zeros(len(features), len(self.bias), device=features.device)
        sensory = rates.index_copy(1, self.input_idx, self.sensory(features))
        leak = torch.sigmoid(self.leak_logit)
        weights = self.edge_weights()
        for _ in range(self.config.steps):
            drive = self.propagate(rates, ablation, weights) + sensory + self.bias
            activation = self.config.saturation * torch.tanh(torch.relu(drive) / self.config.saturation)
            rates = (1 - leak) * rates + leak * activation
        return rates

    def forward(self, features: torch.Tensor, mask: torch.Tensor, ablation: str = 'real'):
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT or mask.shape != (len(features), 7) or not mask.any(dim=1).all():
            raise ValueError('invalid structured observation or legal-action mask')
        motor = self.motor_norm(self.rates(features, ablation).index_select(1, self.output_idx))
        return self.policy(motor).masked_fill(~mask, -torch.inf), torch.tanh(self.value(motor)).squeeze(-1)

    def metadata(self) -> dict:
        return {'model': asdict(self.config), 'graph_sha256': self.graph_sha,
            'neurons': len(self.bias), 'connections': len(self.signs),
            'inputs': len(self.input_idx), 'outputs': len(self.output_idx)}

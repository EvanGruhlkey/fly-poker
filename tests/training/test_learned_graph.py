import torch
from training.data import tiny_graph
from training.model import BrainPolicy


def test_learned_magnitudes_and_per_neuron_leak_receive_gradients():
    model = BrainPolicy(tiny_graph())
    assert model.leak_logit.shape == (5,)
    inputs = torch.rand(3, 118)
    before = model.syn_gain.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    logits, values = model(inputs, torch.ones(3, 7, dtype=torch.bool))
    (logits.square().sum() + values.sum()).backward()
    assert model.syn_gain.grad.abs().sum() > 0
    assert model.leak_logit.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.syn_gain)
    assert torch.equal(model.edge_weights().sign().to(torch.int8), model.signs)
    assert model.rates(inputs).min() >= 0


def test_sparse_edge_and_state_gradients_match_dense_reference():
    from training.sparse import signed_mm
    model = BrainPolicy(tiny_graph())
    weights = model.edge_weights().detach().requires_grad_()
    state = torch.rand(5, 3, requires_grad=True)
    sparse = signed_mm(weights, state, model)
    dense = torch.sparse_csr_tensor(model.csr_indptr, model.csr_indices, weights,
        size=(5, 5)).to_dense() @ state
    sparse_grad = torch.autograd.grad(sparse.square().sum(), (weights, state), retain_graph=True)
    dense_grad = torch.autograd.grad(dense.square().sum(), (weights, state))
    for actual, expected in zip(sparse_grad, dense_grad):
        assert torch.allclose(actual, expected, atol=1e-6)

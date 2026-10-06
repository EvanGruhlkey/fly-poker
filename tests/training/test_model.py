import torch
from training.data import tiny_graph
from training.features import FEATURE_COUNT
from training.model import BrainPolicy

torch.set_num_threads(2)


def test_signed_sparse_propagation_matches_dense_tiny_graph():
    model = BrainPolicy(tiny_graph())
    state = torch.tensor([[.2, .8, .3, .1, .4], [.9, .1, .5, .4, .3]])
    expected = state @ model.adjacency.to_dense().T
    assert torch.allclose(model.propagate(state), expected)
    assert model.propagate(state)[0, 2] < 0


def test_gradient_learning_changes_adapters_without_changing_graph():
    torch.manual_seed(7)
    model = BrainPolicy(tiny_graph())
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    sensory_before = model.sensory.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    features = torch.randn(4, FEATURE_COUNT)
    mask = torch.ones(4, 7, dtype=torch.bool)
    logits, value = model(features, mask)
    loss = -torch.log_softmax(logits, -1)[:, 4].mean() + (value - .7).square().mean()
    loss.backward()
    assert model.sensory.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(sensory_before, model.sensory.weight)
    for name, value in model.named_buffers():
        old = fixed[name]
        if value.layout == torch.sparse_csr:
            assert torch.equal(value.values(), old.values())
            assert torch.equal(value.col_indices(), old.col_indices())
            assert torch.equal(value.crow_indices(), old.crow_indices())
        else:
            assert torch.equal(value, old)
    zero, _ = model(features, mask, ablation='zero')
    assert not torch.allclose(logits, zero)


def test_illegal_actions_have_zero_probability():
    model = BrainPolicy(tiny_graph())
    mask = torch.tensor([[False, True, False, False, False, False, False]])
    logits, _ = model(torch.zeros(1, FEATURE_COUNT), mask)
    assert torch.softmax(logits, -1).tolist() == [[0, 1, 0, 0, 0, 0, 0]]


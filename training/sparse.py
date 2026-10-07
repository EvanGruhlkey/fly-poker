import torch


class SparseMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, state, indptr, indices, transptr, transindices, permutation):
        size = (state.shape[0], state.shape[0])
        matrix = torch.sparse_csr_tensor(indptr, indices, weights, size=size)
        ctx.save_for_backward(weights, state, indptr, indices, transptr, transindices, permutation)
        return torch.sparse.mm(matrix, state)

    @staticmethod
    def backward(ctx, gradient):
        weights, state, indptr, indices, transptr, transindices, permutation = ctx.saved_tensors
        size = (state.shape[0], state.shape[0])
        transpose = torch.sparse_csr_tensor(transptr, transindices, weights[permutation], size=size)
        state_gradient = torch.sparse.mm(transpose, gradient)
        pattern = torch.sparse_csr_tensor(indptr, indices, torch.zeros_like(weights), size=size)
        weight_gradient = torch.sparse.sampled_addmm(pattern, gradient, state.T, beta=0).values()
        return weight_gradient, state_gradient, None, None, None, None, None


def signed_mm(weights, state, structure):
    return SparseMultiply.apply(weights, state, structure.csr_indptr, structure.csr_indices,
        structure.transpose_indptr, structure.transpose_indices, structure.transpose_permutation)

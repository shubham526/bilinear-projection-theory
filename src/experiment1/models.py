import torch
import torch.nn as nn

def construct_theoretical_W_I0(n_dim, I0_indices):
    """
    Constructs the theoretical rank-2 bilinear matrix W_I0 = e_i1*e_i1^T + e_i2*e_i2^T
    where e_k are standard basis vectors (one-hot).
    I0_indices is a tuple like (i1, i2).
    """
    if len(I0_indices) != 2:
        raise ValueError("I0_indices must contain exactly two distinct indices.")
    i1, i2 = I0_indices
    if i1 == i2:
        raise ValueError("Indices in I0_indices must be distinct.")
    if not (0 <= i1 < n_dim and 0 <= i2 < n_dim):
        raise ValueError(f"Indices must be between 0 and {n_dim - 1}.")

    W_I0 = torch.zeros((n_dim, n_dim), dtype=torch.float32)
    W_I0[i1, i1] = 1.0
    W_I0[i2, i2] = 1.0
    return W_I0


class BilinearScorer(nn.Module):
    """A simple module to apply a fixed bilinear matrix W."""

    def __init__(self, W_matrix):
        super().__init__()
        if not isinstance(W_matrix, torch.Tensor):
            W_matrix = torch.tensor(W_matrix, dtype=torch.float32)
        self.W = nn.Parameter(W_matrix, requires_grad=False)  # W is fixed

    def forward(self, query_embed, passage_embed):
        # query_embed: (batch_size, n_dim) or (n_dim)
        # passage_embed: (batch_size, n_dim) or (n_dim)
        if query_embed.ndim == 1:
            query_embed = query_embed.unsqueeze(0)
        if passage_embed.ndim == 1:
            passage_embed = passage_embed.unsqueeze(0)

        # s = q^T W d
        q_W = torch.matmul(query_embed, self.W)  # (batch_size, n_dim)
        scores = torch.sum(q_W * passage_embed, dim=1)  # (batch_size,)
        return scores


class WeightedDotProductModel(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        # Trainable weight vector v
        self.v_weights = nn.Parameter(torch.ones(n_dim))  # Initialize with ones or random
        # nn.init.xavier_uniform_(self.v_weights.unsqueeze(0)) # Alternative initialization

    def forward(self, query_embed, passage_embed):
        # query_embed: (batch_size, n_dim) or (n_dim)
        # passage_embed: (batch_size, n_dim) or (n_dim)
        if query_embed.ndim == 1:
            query_embed = query_embed.unsqueeze(0)
        if passage_embed.ndim == 1:
            passage_embed = passage_embed.unsqueeze(0)

        # s_v(q,d) = sum_j v_j * q_j * d_j
        # Element-wise product: v * q * d, then sum over dimensions
        weighted_q_d_product = self.v_weights * query_embed * passage_embed
        scores = torch.sum(weighted_q_d_product, dim=1)  # (batch_size,)
        return scores


if __name__ == '__main__':
    # Example Usage
    n_test = 4
    I0_test = (0, 2)
    W = construct_theoretical_W_I0(n_test, I0_test)
    print(f"Theoretical W_I0 for n={n_test}, I0={I0_test}:\n{W}")

    bilinear_model = BilinearScorer(W)
    wdp_model = WeightedDotProductModel(n_test)

    q_sample = torch.tensor([1., -1., 1., -1.], dtype=torch.float32)
    d_sample1 = torch.tensor([1., 1., 1., 1.], dtype=torch.float32)  # Agrees on 0, disagrees on 2
    d_sample2 = torch.tensor([1., -1., -1., -1.], dtype=torch.float32)  # Agrees on 0, 1, 3. Disagrees on 2.

    score_bilinear = bilinear_model(q_sample, d_sample1)
    print(f"Bilinear score (q, d1): {score_bilinear.item()}")  # Expected: q0*d0 + q2*d2 = 1*1 + 1*1 = 2

    score_wdp = wdp_model(q_sample, d_sample1)
    print(f"WDP score (q, d1) (untrained v): {score_wdp.item()}")

    # Batch test
    q_batch = torch.randn(3, n_test)
    d_batch = torch.randn(3, n_test)
    print(f"Bilinear batch scores: {bilinear_model(q_batch, d_batch)}")
    print(f"WDP batch scores: {wdp_model(q_batch, d_batch)}")
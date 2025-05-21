# models.py
import torch
import torch.nn as nn


class DotProductModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # No trainable parameters for pure dot product with fixed embeddings
        pass

    def forward(self, query_embed, passage_embed):
        # query_embed: (batch_size, embedding_dim)
        # passage_embed: (batch_size, embedding_dim)
        # Output: (batch_size,)
        return torch.sum(query_embed * passage_embed, dim=1)


class WeightedDotProductModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.weights = nn.Parameter(torch.ones(self.embedding_dim))  # Initialize with ones

    def forward(self, query_embed, passage_embed):
        # Output: (batch_size,)
        return torch.sum(self.weights * query_embed * passage_embed, dim=1)


class LowRankBilinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.rank = config["rank"]

        # P and Q matrices for W = P @ Q.T
        # Explicitly set dtype to float32
        self.P = nn.Parameter(torch.Tensor(self.embedding_dim, self.rank).to(torch.float32))
        self.Q = nn.Parameter(torch.Tensor(self.embedding_dim, self.rank).to(torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        # Ensure initialization also uses float32
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)

    def forward(self, query_embed, passage_embed):
        # Ensure input tensors are float32
        if query_embed.dtype != torch.float32:
            query_embed = query_embed.to(torch.float32)
        if passage_embed.dtype != torch.float32:
            passage_embed = passage_embed.to(torch.float32)

        # Compute s = (P^T q)^T (Q^T d) which is equivalent to q^T P Q^T d
        # More efficient factorized computation:
        query_proj = torch.matmul(query_embed, self.P)  # (batch_size, rank)
        passage_proj = torch.matmul(passage_embed, self.Q)  # (batch_size, rank)
        scores = torch.sum(query_proj * passage_proj, dim=1)  # (batch_size,)
        return scores

class FullRankBilinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.W = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim).to(torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, query_embed, passage_embed):
        # Ensure input tensors are float32
        if query_embed.dtype != torch.float32:
            query_embed = query_embed.to(torch.float32)
        if passage_embed.dtype != torch.float32:
            passage_embed = passage_embed.to(torch.float32)

        # Compute s = q^T W d
        q_W = torch.matmul(query_embed, self.W)  # (batch_size, embedding_dim)
        scores = torch.sum(q_W * passage_embed, dim=1)  # (batch_size,)
        return scores


def get_model(model_name, config_params):
    model_type = config_params["type"]
    if model_type == "dot_product":
        return DotProductModel(config_params)
    elif model_type == "weighted_dot_product":
        return WeightedDotProductModel(config_params)
    elif model_type == "low_rank_bilinear":
        return LowRankBilinearModel(config_params)
    elif model_type == "full_rank_bilinear":
        return FullRankBilinearModel(config_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
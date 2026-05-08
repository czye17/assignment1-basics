from einops import rearrange

import numpy as np
import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device:torch.device | None = None, dtype: torch.dtype | None =None):
        super().__init__()
        self.embedding_dim = embedding_dim
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.nn.init.trunc_normal_(torch.zeros((num_embeddings, embedding_dim), dtype=dtype), mean=0.0, std=1, a=-3, b=3)
        self.weights = torch.nn.Parameter(weights)
        if device is not None:
            self.weights.to(device)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

import pdb

import numpy as np
import torch

from einops import reduce, einsum, rearrange


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = 'cpu' if device is None else device
        self.norms = torch.nn.Parameter(torch.ones(d_model)).to(self.device)
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(self.device)
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x ** 2, 'b s d -> b s', 'sum') / self.d_model + self.eps)
        result = einsum(x, self.norms, 'b s d, d -> b s d')
        result = einsum(result, 1 / rms, 'b s d, b s -> b s d')

        return result.to(in_dtype)


if __name__ == '__main__':
    d_model = 5
    normalizer = RMSNorm(d_model)
    
    x1 = rearrange(torch.ones(d_model), 'd -> 1 1 d')
    y1 = normalizer(x1)
    print(f'---- Test 1: One Vector ----')
    print(f'In:  {x1}')
    print(f'Out: {y1}')

    x2 = rearrange(torch.ones(d_model) * 87, 'd -> 1 1 d')
    y2 = normalizer(x2)
    print(f'---- Test 2: Scaled One Vector ----')
    print(f'In:  {x2}')
    print(f'Out: {y2}')

    x3 = torch.randn(3, 3, d_model) * 10
    y3 = normalizer(x3)
    print(f'---- Test 3: Scaled One Vector ----')
    print(f'In:  {x3}')
    print(f'Out: {y3}')



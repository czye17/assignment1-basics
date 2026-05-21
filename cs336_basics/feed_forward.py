from einops import einsum

import numpy as np
import torch


def silu(x: torch.Tensor):
    return einsum(x, torch.sigmoid(x), '..., ... -> ...')


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()
        self.dtype = torch.float32 if dtype is None else dtype
        self.d_model = d_model
        self.d_ff = d_ff

        device = 'cpu' if device is None else device
        std = np.sqrt(2/(d_model + d_ff))
        self.weights1 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((d_ff, d_model), dtype=self.dtype), mean=0.0, std=std, a=-3*std, b=3*std)).to(device)
        self.weights2 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((d_model, d_ff), dtype=self.dtype), mean=0.0, std=std, a=-3*std, b=3*std)).to(device)
        self.weights3 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((d_ff, d_model), dtype=self.dtype), mean=0.0, std=std, a=-3*std, b=3*std)).to(device)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weights1.device)
        w1_x = einsum(x, self.weights1, '... d_model, d_ff d_model -> ... d_ff')
        silu_out = silu(w1_x)
        w3_x = einsum(x, self.weights3, '... d_model, d_ff d_model -> ... d_ff')
        swiglu = einsum(silu_out, w3_x, '..., ... -> ...')
        result = einsum(swiglu, self.weights2, '... d_ff, d_model d_ff -> ... d_model')
        return result

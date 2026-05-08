from einops import einsum

import numpy as np
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = np.sqrt(2/(in_features + out_features))
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.nn.init.trunc_normal_(torch.zeros((out_features, in_features), dtype=dtype), mean=0.0, std=std, a=-3*std, b=3*std)
        self.weights = torch.nn.Parameter(weights)
        self.device = 'cpu' if device is None else device
        self.weights.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.to(self.device)
        return einsum(x, self.weights, '... d_in, d_out d_in -> ... d_out')


if __name__ == '__main__':
    model = Linear(2, 3)
    state_dict = model.state_dict()
    for k in state_dict.keys():
        print(k)


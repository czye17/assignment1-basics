import pdb
from einops import rearrange, einsum

import torch


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        assert(d_k % 2 == 0)
        super().__init__()
        device = 'cpu' if device is None else device
        self.theta = theta
        self.d_k = d_k
        angles = torch.Tensor([[i/(theta ** (2*k/d_k)) for k in range(d_k // 2)] for i in range(max_seq_len)]).to(device)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotation = rearrange(torch.stack([cos, - sin, sin, cos]), '(r c) ... -> ... r c', r=2, c=2)
        self.register_buffer('rotation', rotation, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # pdb.set_trace()
        x = x.to(self.rotation.device)
        token_positions = token_positions.to(self.rotation.device)
        x = rearrange(x, '... s (k k2) -> ... s k k2', k2 = 2)
        order_rotation = self.rotation[token_positions]
        result = einsum(order_rotation, x, '... s k r c, ... s k c -> ... s k r')
        return rearrange(result, '... s k r -> ... s (k r)')

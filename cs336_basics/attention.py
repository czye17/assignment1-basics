import pdb

from functools import partial
import numpy as np
import torch
from einops import einsum, rearrange
from cs336_basics.rope import RoPE
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.feed_forward import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


MASK_MAP = np.vectorize(lambda x: 0.0 if x else float('-inf'))


def softmax(x: torch.Tensor, dim: int):
    x = x - torch.amax(x, axis=dim, keepdim=True)
    x = torch.exp(x)
    x = x / torch.sum(x, axis=dim, keepdim=True)
    return x


def attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None):
    # pdb.set_trace()
    d_k = queries.shape[-1]
    att_raw = einsum(queries, keys, '... q d, ... k d -> ... q k') / (d_k ** 0.5)
    mask_map = torch.Tensor(MASK_MAP(mask), device=att_raw.device)
    masked_att = att_raw + mask_map if mask is not None else att_raw
    att = softmax(masked_att, dim=-1)
    result = einsum(att, values, '... q k, ... k d -> ... q d')
    return result


class MultiHeadSelfAttentionNoRoPE(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, device:torch.device | None=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        std = 1/(d_model ** 0.5)
        device = 'cpu' if device is None else device
        proj = torch.nn.init.trunc_normal_(torch.zeros((d_model, 3 * d_model)), mean=0.0, std=std, a=-3*std, b=3*std).to(device)
        self.proj = torch.nn.Parameter(proj)
        out_proj = torch.nn.init.trunc_normal_(torch.zeros((d_model, d_model)), mean=0.0, std=std, a=-3*std, b=3*std).to(device)
        self.out_proj = torch.nn.Parameter(out_proj)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        projections = einsum(x, self.proj, '... s d, d d3 -> ... s d3')
        qkv = rearrange(projections,  '... s (type head d_head) -> type ... head s d_head', type=3, head=self.n_heads)
        q, k, v = qkv

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        attn_out = attention(q, k, v, mask=mask)
        attn_concat = rearrange(attn_out, '... head s d_head -> ... s (head d_head)')
        result = einsum(attn_concat, self.out_proj, '... s d_in, d_in d_out -> ... s d_out')
        return result


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float=0.0, device:torch.device | None=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask, persistent=False)
        self.rope = RoPE(theta, self.d_head, max_seq_len, device=device)
        std = 1/(d_model ** 0.5)
        device = 'cpu' if device is None else device
        proj = torch.nn.init.trunc_normal_(torch.zeros((d_model, 3 * d_model)), mean=0.0, std=std, a=-3*std, b=3*std).to(device)
        self.proj = torch.nn.Parameter(proj)
        out_proj = torch.nn.init.trunc_normal_(torch.zeros((d_model, d_model)), mean=0.0, std=std, a=-3*std, b=3*std).to(device)
        self.out_proj = torch.nn.Parameter(out_proj)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        projections = einsum(x, self.proj, '... s d, d d3 -> ... s d3')
        qkv = rearrange(projections,  '... s (type head d_head) -> type ... head s d_head', type=3, head=self.n_heads)
        q, k, v = qkv

        token_positions = torch.arange(seq_len).to(x.device) if token_positions is None else token_positions
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        mask = self.mask[:seq_len, :seq_len]
        attn_out = attention(q, k, v, mask=mask)
        attn_concat = rearrange(attn_out, '... head s d_head -> ... s (head d_head)')
        result = einsum(attn_concat, self.out_proj, '... s d_in, d_in d_out -> ... s d_out')
        return result


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta=theta, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        att0 = self.ln1(x)
        token_positions = torch.arange(seq_len).to(x.device)
        att1 = self.attn(att0, token_positions)
        x = x + att1

        mlp0 = self.ln2(x)
        mlp1 = self.ffn(mlp0)
        x = x + mlp1

        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, vocab_size: int, num_layers: int, theta: float, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.tf_blocks = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
    

    def forward(self, x):
        x0 = self.embedding(x)
        for block in self.tf_blocks:
            x0 = block(x0)
        x1 = self.ln(x0)
        logits = self.linear(x1)
        return logits
    

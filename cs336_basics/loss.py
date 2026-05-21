import pdb
from einops import reduce
import torch

from cs336_basics.attention import softmax


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # pdb.set_trace()
    seq_len = targets.shape[-1]
    logits = logits - torch.amax(logits, axis=-1, keepdim=True)
    norms = reduce(torch.exp(logits), '... v -> ...', 'sum')
    log_softmax = logits[torch.arange(seq_len), targets] - torch.log(norms)
    return torch.mean(- log_softmax)

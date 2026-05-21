import pdb

from einops import rearrange
import numpy as np
import torch


def data_loader(dataset, batch_size: int, context_length: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    device = 'cpu' if device is None else device
    valid_starting_indices = len(dataset) - (context_length + 1)
    starting_indices = torch.randint(low=0, high=valid_starting_indices + 1, size=(batch_size, 1))
    xs = torch.Tensor(np.array([dataset[i:i+context_length] for i in starting_indices])).to(device)
    ys = torch.Tensor(np.array([dataset[i+1:i+context_length+1] for i in starting_indices])).to(device)
    return xs, ys


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iter: int, out):
    checkpoint = {
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'iter': iter
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    iter = checkpoint['iter']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    return iter
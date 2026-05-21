import pdb

from einops import reduce
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
import math


def gradient_clip(parameters, max_l2_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    grad_norm = 0
    for p in params:
        grad_norm += reduce(p.grad ** 2, '... -> ', 'sum')
    
    grad_norm = torch.sqrt(grad_norm)
    if grad_norm >= max_l2_norm:
        for p in params:
            p.grad *= max_l2_norm / (grad_norm + eps)


def lr_cosine_schedule(t, lr_max, lr_min, t_warm, t_cos):
    assert (t >= 0)
    if t < t_warm:
        return (t / t_warm) * lr_max
    elif t <= t_cos:
        return lr_min + 0.5 * (1 + np.cos(((t - t_warm)/(t_cos - t_warm)) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                
                t = state.get('t', 1)
                m = state.get('m', torch.zeros(p.data.shape))
                v = state.get('v', torch.zeros(p.data.shape))

                lr_t = lr * np.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= lr * weight_decay * p.data # apply weight decay
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss

    

if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    lr = 1e3
    print(f'learning rate:{lr}')
    opt = SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

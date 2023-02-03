import torch


@torch.jit.script
def factorial(n: torch.Tensor):
    """
    A torch-only factorial function.
    """
    return torch.exp(torch.lgamma(n+1))

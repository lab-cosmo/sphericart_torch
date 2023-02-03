import torch


@torch.jit.script
def phi_dependent_recursions(l_max: int, x: torch.Tensor, y: torch.Tensor):
    """
    Recursive evaluation of c_m = r_(xy)^m * cos(m*phi) and s_m = r_(xy)^m * sin(m*phi).
    """
    
    c = torch.empty((x.shape[0], l_max+1), device=x.device, dtype=x.dtype)
    s = torch.empty((x.shape[0], l_max+1), device=x.device, dtype=x.dtype)
    c[:, 0] = 1.0
    s[:, 0] = 0.0
    for m in range(1, l_max+1):
        c[:, m] = c[:, m-1].clone()*x - s[:, m-1].clone()*y
        s[:, m] = s[:, m-1].clone()*x + c[:, m-1].clone()*y

    Phi = torch.cat([
        s[:, 1:].flip(dims=[1]),
        torch.sqrt(torch.tensor([1.0/2.0], device=x.device, dtype=x.dtype))*torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
        c[:, 1:]
    ], dim=-1)

    return Phi


@torch.jit.script
def phi_dependent_recursions_derivatives(Phi: torch.Tensor):

    l_max = Phi.shape[1] - 1
    if l_max == 0:  # Special case where the below stacking would not work.
        grad_Phi = [
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype)
        ]
        return grad_Phi

    m = torch.arange(-l_max, l_max+1)

    grad_Phi = [
        m*torch.cat([
            Phi[:, 1:l_max],
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            torch.ones((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            Phi[:, l_max+1:2*l_max],
        ], dim=-1),
        m*torch.cat([
            Phi[:, l_max+1:2*l_max].flip(dims=[1]),
            torch.ones((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            torch.zeros((Phi.shape[0], 1), device=Phi.device, dtype=Phi.dtype),
            -Phi[:, 1:l_max].flip(dims=[1]),
        ], dim=-1)
    ]

    return grad_Phi

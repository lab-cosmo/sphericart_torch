import torch
from .utils import factorial
from typing import List


@torch.jit.script
def combine_into_spherical_harmonics(Qlm: List[torch.Tensor], Phi: torch.Tensor, r: torch.Tensor):

    l_max = (Phi.shape[1] - 1) // 2
    Y = []
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        Y.append(
            torch.pow(-1, m) * torch.sqrt(torch.tensor([2.0], dtype=r.dtype, device=r.device))
            * torch.sqrt((2*l+1)/(4.0*2.0*torch.acos(torch.zeros(1, device=r.device)))*factorial(l-abs_m)/factorial(l+abs_m))
            * Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
        )
    return Y


@torch.jit.script
def combine_into_spherical_harmonics_gradients(
        Qlm: List[torch.Tensor], 
        Phi: torch.Tensor, 
        grad_Qlm: List[List[torch.Tensor]], 
        grad_Phi: List[torch.Tensor], 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor, 
        r: torch.Tensor
    ):

    l_max = (Phi.shape[1] - 1) // 2
    # Fill the output tensor list:
    grad_Y: List[List[torch.Tensor]] = []

    grad_Y.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        grad_Y[0].append(
            torch.pow(-1, m) * torch.sqrt(torch.tensor([2.0], dtype=r.dtype, device=r.device))
            * torch.sqrt((2*l+1)/(4.0*2.0*torch.acos(torch.zeros(1, device=r.device)))*factorial(l-abs_m)/factorial(l+abs_m))
            * ( Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            * (-l*x*r**(-l-2)).unsqueeze(dim=-1)
            +
            grad_Qlm[0][l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            +
            Qlm[l][:, abs_m]
            * grad_Phi[0][:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            )
        )
    
    grad_Y.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        grad_Y[1].append(
            torch.pow(-1, m) * torch.sqrt(torch.tensor([2.0], dtype=r.dtype, device=r.device))
            * torch.sqrt((2*l+1)/(4.0*2.0*torch.acos(torch.zeros(1, device=r.device)))*factorial(l-abs_m)/factorial(l+abs_m))
            * ( Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            * (-l*y*r**(-l-2)).unsqueeze(dim=-1)
            +
            grad_Qlm[1][l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            +
            Qlm[l][:, abs_m]
            * grad_Phi[1][:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            )
        )

    grad_Y.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        grad_Y[2].append(
            torch.pow(-1, m) * torch.sqrt(torch.tensor([2.0], dtype=r.dtype, device=r.device))
            * torch.sqrt((2*l+1)/(4.0*2.0*torch.acos(torch.zeros(1, device=r.device)))*factorial(l-abs_m)/factorial(l+abs_m))
            * ( Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            * (-l*z*r**(-l-2)).unsqueeze(dim=-1)
            +
            grad_Qlm[2][l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            )
        )

    return grad_Y

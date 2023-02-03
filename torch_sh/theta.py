import torch
from typing import List


@torch.jit.script
def modified_associated_legendre_polynomials(l_max: int, z: torch.Tensor, r2: torch.Tensor):

    """
    Calculates Q_l^m = P_l^m * r^l / r_(xy)^m , where P_l^m is an associated Legendre polynomial.
    This construction simplifies the calculation in Cartesian coordinates and it avoids the
    numerical instabilities in the derivatives for points on the z axis. The implementation is
    based on the standard recurrence relations for P_l^m.
    """

    Qlm = []
    for l in range(l_max+1):
        Qlm.append(
            torch.empty((l+1, z.shape[0]), dtype = z.dtype, device = z.device)
        )

    Qlm[0][0] = 1.0
    for m in range(1, l_max+1):
        Qlm[m][m] = -(2*m-1)*Qlm[m-1][m-1].clone()
        Qlm[m][m-1] = (2*m-1)*z*Qlm[m-1][m-1].clone()
    for m in range(l_max-1):
        for l in range(m+2, l_max+1):
            Qlm[l][m] = ((2*l-1)*z*Qlm[l-1][m].clone()-(l+m-1)*Qlm[l-2][m].clone()*r2)/(l-m)

    Qlm = [Qlm_l.swapaxes(0, 1) for Qlm_l in Qlm]
    return Qlm


@torch.jit.script
def modified_associated_legendre_polynomials_derivatives(Qlm: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor):

    l_max = len(Qlm) - 1
    d_Qlm_d_x = []
    d_Qlm_d_y = []
    d_Qlm_d_z = []

    for l in range(l_max+1):
        d_Qlm_d_x.append(torch.zeros_like(Qlm[l]))
        d_Qlm_d_y.append(torch.zeros_like(Qlm[l]))
        d_Qlm_d_z.append(torch.zeros_like(Qlm[l]))
        if l == 0: continue
        m = torch.arange(0, l)

        d_Qlm_d_x[l][:, :-2] = x.unsqueeze(dim=-1)*Qlm[l-1][:, 1:]
        d_Qlm_d_y[l][:, :-2] = y.unsqueeze(dim=-1)*Qlm[l-1][:, 1:]
        d_Qlm_d_z[l][:, :-1] = (l+m)*Qlm[l-1]

    grad_Qlm = [d_Qlm_d_x, d_Qlm_d_y, d_Qlm_d_z]

    return grad_Qlm

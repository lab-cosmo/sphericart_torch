import torch

@torch.jit.script
def modified_associated_legendre_polynomials(l_max: int, z: torch.Tensor, r: torch.Tensor):

    """
    Calculates P_l^m * r^l / r_(xy)^m , where P_l^m is an associated Legendre polynomial.
    This construction simplifies the calculation in Cartesian coordinates and it avoids the
    numerical instabilities in the derivatives for points on the z axis. The implementation is
    based on the standard recurrence relations for P_l^m.
    """

    q = []
    for l in range(l_max+1):
        q.append(
            torch.empty((l+1, r.shape[0]), dtype = r.dtype, device = r.device)
        )

    q[0][0] = 1.0
    for m in range(1, l_max+1):
        q[m][m] = -(2*m-1)*q[m-1][m-1].clone()
    for m in range(l_max):
        q[m+1][m] = (2*m+1)*z*q[m][m].clone()
    for m in range(l_max-1):
        for l in range(m+2, l_max+1):
            q[l][m] = ((2*l-1)*z*q[l-1][m].clone()-(l+m-1)*q[l-2][m].clone()*r**2)/(l-m)

    q = [q_l.swapaxes(0, 1) for q_l in q]
    return q


def modified_associated_legendre_polynomials_derivatives(Qlm, x, y):

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

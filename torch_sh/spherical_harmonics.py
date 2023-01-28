import torch
    
@torch.jit.script
def factorial(n: torch.Tensor):
    """
    A torch-only factorial function.
    """
    return torch.exp(torch.lgamma(n+1))


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


@torch.jit.script
def spherical_harmonics(l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

    """
    Calculates all the real spherical harmonics up to order l_max.

    Inputs:
    l_max: maximum order of the output spherical harmonics
    x, y, z: 1-D tensors, corresponding to the points in 3D Cartesian space for which
    the spherical harmonics need to be calculated. All three have dimensions [n_points].

    Output:
    output: list of torch.Tensors, corresponding to each l from 0 to l_max, containing
    the spherical harmonics. output[l] has shape [n_points, 2*l+1].
    """

    # Precomputations:
    r = torch.sqrt(x**2+y**2+z**2)
    sqrt_2 = torch.sqrt(torch.tensor([2.0], device=r.device, dtype=r.dtype))
    one_over_sqrt_2 = 1.0/sqrt_2
    pi = 2.0 * torch.acos(torch.zeros(1, device=r.device))

    # theta-dependent component of the spherical harmonics:
    Qlm = modified_associated_legendre_polynomials(l_max, z, r)
    
    # Recursive evaluation of c_m = r_(xy)^m * cos(m*phi) and s_m = r_(xy)^m * sin(m*phi):
    c = torch.empty((r.shape[0], l_max+1), device=r.device, dtype=r.dtype)
    s = torch.empty((r.shape[0], l_max+1), device=r.device, dtype=r.dtype)
    c[:, 0] = 1.0
    s[:, 0] = 0.0
    for m in range(1, l_max+1):
        c[:, m] = c[:, m-1]*x - s[:, m-1]*y
        s[:, m] = s[:, m-1]*x + c[:, m-1]*y

    # phi-dependent component of the spherical harmonics:
    Phi = torch.cat([
        s[:, 1:].flip(dims=[1]),
        one_over_sqrt_2*torch.ones((r.shape[0], 1), device=r.device, dtype=r.dtype),
        c[:, 1:]
    ], dim=-1)

    # Fill the output tensor list:
    output = []
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        output.append(
            torch.pow(-1, m) * sqrt_2
            * torch.sqrt((2*l+1)/(4*pi)*factorial(l-abs_m)/factorial(l+abs_m))
            * Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
        )

    return output

        


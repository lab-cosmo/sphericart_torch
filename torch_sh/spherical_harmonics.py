import torch
from typing import List
from .theta import modified_associated_legendre_polynomials, modified_associated_legendre_polynomials_derivatives
from .phi import phi_dependent_recursions, phi_dependent_recursions_derivatives
from .utils import factorial


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
    
    # phi-dependent component of the spherical harmonics:
    Phi = phi_dependent_recursions(l_max, x, y)

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


def spherical_harmonics_gradients(sh_object: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

    l_max = len(sh_object) - 1
    x_grad = []
    y_grad = []
    z_grad = []

    for l in range(l_max+1):
        x_grad_l = []
        y_grad_l = []
        z_grad_l = []

        for m in range(-l, l+1):

            x_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = x,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            x_grad_l.append(x_grad_l_m)

            y_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = y,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            y_grad_l.append(y_grad_l_m)

            z_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = z,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            z_grad_l.append(z_grad_l_m)

        x_grad_l = torch.stack(x_grad_l, dim=-1)
        y_grad_l = torch.stack(y_grad_l, dim=-1)
        z_grad_l = torch.stack(z_grad_l, dim=-1)

        x_grad.append(x_grad_l)
        y_grad.append(y_grad_l)
        z_grad.append(z_grad_l)

    gradients = [x_grad, y_grad, z_grad]
    return gradients


def spherical_harmonics_custom_gradients(l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

    r = torch.sqrt(x**2+y**2+z**2)
    sqrt_2 = torch.sqrt(torch.tensor([2.0], device=r.device, dtype=r.dtype))
    one_over_sqrt_2 = 1.0/sqrt_2
    pi = 2.0 * torch.acos(torch.zeros(1, device=r.device))

    # theta-dependent component of the spherical harmonics:
    Qlm = modified_associated_legendre_polynomials(l_max, z, r)
    grad_Qlm = modified_associated_legendre_polynomials_derivatives(Qlm, x, y)
    
    # phi-dependent component of the spherical harmonics:
    Phi = phi_dependent_recursions(l_max, x, y)
    grad_Phi = phi_dependent_recursions_derivatives(Phi)

    # Fill the output tensor list:
    gradients = []

    gradients.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        gradients[0].append(
            torch.pow(-1, m) * sqrt_2
            * torch.sqrt((2*l+1)/(4*pi)*factorial(l-abs_m)/factorial(l+abs_m))
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
    
    gradients.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        gradients[1].append(
            torch.pow(-1, m) * sqrt_2
            * torch.sqrt((2*l+1)/(4*pi)*factorial(l-abs_m)/factorial(l+abs_m))
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

    gradients.append([])
    for l in range(l_max+1):
        m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
        abs_m = torch.abs(m)
        gradients[2].append(
            torch.pow(-1, m) * sqrt_2
            * torch.sqrt((2*l+1)/(4*pi)*factorial(l-abs_m)/factorial(l+abs_m))
            * ( Qlm[l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            * (-l*z*r**(-l-2)).unsqueeze(dim=-1)
            +
            grad_Qlm[2][l][:, abs_m]
            * Phi[:, l_max-l:l_max+l+1]
            / (r**l).unsqueeze(dim=-1)
            )
        )

    return gradients

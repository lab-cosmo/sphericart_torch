import torch
from typing import List
from .theta import modified_associated_legendre_polynomials, modified_associated_legendre_polynomials_derivatives
from .phi import phi_dependent_recursions, phi_dependent_recursions_derivatives
from .utils import factorial


class SphericalHarmonics(torch.autograd.Function):

    # Shared variables to be initialized:
    sqrt_2 = None
    one_over_sqrt_2 = None
    pi = None
    forward_gradients = None

    @staticmethod
    def initialize(device, forward_gradients=False):
        SphericalHarmonics.sqrt_2 = torch.sqrt(torch.tensor([2.0], device=device))
        SphericalHarmonics.one_over_sqrt_2 = 1.0/SphericalHarmonics.sqrt_2
        SphericalHarmonics.pi = 2.0 * torch.acos(torch.zeros(1, device=device))
        SphericalHarmonics.forward_gradients = forward_gradients


    @staticmethod
    def compute(l_max, x, y, z):
        if SphericalHarmonics.forward_gradients:
            r = torch.sqrt(x**2+y**2+z**2)
            Qlm = modified_associated_legendre_polynomials(l_max, z, r)
            Phi = phi_dependent_recursions(l_max, x, y)
            Y = SphericalHarmonics.combine_into_spherical_harmonics(Qlm, Phi, r)
            grad_Y = SphericalHarmonics.spherical_harmonics_custom_gradients(Qlm, Phi, x, y, z, r)
            return Y, grad_Y
        else:
            Y = SphericalHarmonics.apply(l_max, x, y, z)
            return Y


    @staticmethod
    def forward(ctx, l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

        r = torch.sqrt(x**2+y**2+z**2)
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        Phi = phi_dependent_recursions(l_max, x, y)
        Y = SphericalHarmonics.combine_into_spherical_harmonics(Qlm, Phi, r)

        return Y


    @staticmethod
    def backward(l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        raise NotImplementedError
        return None


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


    def combine_into_spherical_harmonics(Qlm, Phi, r):

        l_max = (Phi.shape[1] - 1) // 2
        Y = []
        for l in range(l_max+1):
            m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
            abs_m = torch.abs(m)
            Y.append(
                torch.pow(-1, m) * SphericalHarmonics.sqrt_2
                * torch.sqrt((2*l+1)/(4*SphericalHarmonics.pi)*factorial(l-abs_m)/factorial(l+abs_m))
                * Qlm[l][:, abs_m]
                * Phi[:, l_max-l:l_max+l+1]
                / (r**l).unsqueeze(dim=-1)
            )
        return Y


    def spherical_harmonics_custom_gradients(Qlm, Phi, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, r: torch.Tensor):

        l_max = (Phi.shape[1] - 1) // 2

        # theta-dependent component of the spherical harmonics:
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        grad_Qlm = modified_associated_legendre_polynomials_derivatives(Qlm, x, y)
        
        # phi-dependent component of the spherical harmonics:
        Phi = phi_dependent_recursions(l_max, x, y)
        grad_Phi = phi_dependent_recursions_derivatives(Phi)

        grad_Y = SphericalHarmonics.combine_into_spherical_harmonics_gradients(Qlm, Phi, grad_Qlm, grad_Phi, x, y, z, r)
        return grad_Y

        
    def combine_into_spherical_harmonics_gradients(Qlm, Phi, grad_Qlm, grad_Phi, x, y, z, r):

        l_max = (Phi.shape[1] - 1) // 2
        # Fill the output tensor list:
        grad_Y = []

        grad_Y.append([])
        for l in range(l_max+1):
            m = torch.tensor(list(range(-l, l+1)), dtype=torch.long, device=r.device)
            abs_m = torch.abs(m)
            grad_Y[0].append(
                torch.pow(-1, m) * SphericalHarmonics.sqrt_2
                * torch.sqrt((2*l+1)/(4*SphericalHarmonics.pi)*factorial(l-abs_m)/factorial(l+abs_m))
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
                torch.pow(-1, m) * SphericalHarmonics.sqrt_2
                * torch.sqrt((2*l+1)/(4*SphericalHarmonics.pi)*factorial(l-abs_m)/factorial(l+abs_m))
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
                torch.pow(-1, m) * SphericalHarmonics.sqrt_2
                * torch.sqrt((2*l+1)/(4*SphericalHarmonics.pi)*factorial(l-abs_m)/factorial(l+abs_m))
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

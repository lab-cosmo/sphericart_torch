import torch
from typing import List
from .theta import modified_associated_legendre_polynomials, modified_associated_legendre_polynomials_derivatives
from .phi import phi_dependent_recursions, phi_dependent_recursions_derivatives
from .combine import combine_into_spherical_harmonics, combine_into_spherical_harmonics_gradients


class SphericalHarmonics(torch.autograd.Function):

    """
    # Shared variables to be initialized:
    sqrt_2 = None
    one_over_sqrt_2 = None
    pi = None
    """
    forward_gradients = None

    @staticmethod
    def initialize(device, forward_gradients=False):
        """
        SphericalHarmonics.sqrt_2 = torch.sqrt(torch.tensor([2.0], device=device))
        SphericalHarmonics.one_over_sqrt_2 = 1.0/SphericalHarmonics.sqrt_2
        SphericalHarmonics.pi = 2.0 * torch.acos(torch.zeros(1, device=device))
        """
        SphericalHarmonics.forward_gradients = forward_gradients


    @staticmethod
    def compute(l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        if SphericalHarmonics.forward_gradients:
            r = torch.sqrt(x**2+y**2+z**2)
            Qlm = modified_associated_legendre_polynomials(l_max, z, r)
            Phi = phi_dependent_recursions(l_max, x, y)
            Y = combine_into_spherical_harmonics(Qlm, Phi, r)
            grad_Y = SphericalHarmonics.spherical_harmonics_custom_gradients(Qlm, Phi, x, y, z, r)
            return Y, grad_Y
        else:
            Y = SphericalHarmonics.apply(l_max, x, y, z)
            return Y


    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

        r = torch.sqrt(x**2+y**2+z**2)
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        Phi = phi_dependent_recursions(l_max, x, y)
        Y = combine_into_spherical_harmonics(Qlm, Phi, r)

        ctx.l_max = l_max
        ctx.Qlm = Qlm
        ctx.Phi = Phi
        ctx.r = r
        ctx.save_for_backward(x, y, z)

        return tuple(Y)


    @staticmethod
    def backward(ctx, *grad_output):

        # theta-dependent component of the spherical harmonics:
        x, y, z = ctx.saved_tensors
        Qlm = ctx.Qlm
        grad_Qlm = modified_associated_legendre_polynomials_derivatives(Qlm, x, y)
        
        # phi-dependent component of the spherical harmonics:
        Phi = ctx.Phi
        grad_Phi = phi_dependent_recursions_derivatives(Phi)

        r = ctx.r
        grad_Y = combine_into_spherical_harmonics_gradients(Qlm, Phi, grad_Qlm, grad_Phi, x, y, z, r)

        l_max = ctx.l_max
        d_input_d_x = torch.sum(
            torch.cat(
                [grad_output[l]*grad_Y[0][l] for l in range(l_max+1)], 
                dim = 1
            ),
            dim = 1
        )
        d_input_d_y = torch.sum(
            torch.cat(
                [grad_output[l]*grad_Y[1][l] for l in range(l_max+1)], 
                dim = 1
            ),
            dim = 1
        )
        d_input_d_z = torch.sum(
            torch.cat(
                [grad_output[l]*grad_Y[2][l] for l in range(l_max+1)], 
                dim = 1
            ),
            dim = 1
        )

        return None, d_input_d_x, d_input_d_y, d_input_d_z


    @torch.jit.script
    def spherical_harmonics_custom_gradients(Qlm: List[torch.Tensor], Phi: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, r: torch.Tensor):

        l_max = (Phi.shape[1] - 1) // 2

        # theta-dependent component of the spherical harmonics:
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        grad_Qlm = modified_associated_legendre_polynomials_derivatives(Qlm, x, y)
        
        # phi-dependent component of the spherical harmonics:
        Phi = phi_dependent_recursions(l_max, x, y)
        grad_Phi = phi_dependent_recursions_derivatives(Phi)

        grad_Y = combine_into_spherical_harmonics_gradients(Qlm, Phi, grad_Qlm, grad_Phi, x, y, z, r)
        return grad_Y


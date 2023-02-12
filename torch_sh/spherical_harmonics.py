import torch
from typing import List
from .theta import modified_associated_legendre_polynomials, modified_associated_legendre_polynomials_derivatives
from .phi import phi_dependent_recursions, phi_dependent_recursions_derivatives
from .combine import combine_into_spherical_harmonics, combine_into_spherical_harmonics_gradients
from .prefactors import compute_prefactors


class SphericalHarmonics:

    def __init__(self, l_max, device):
        self.prefactors = compute_prefactors(l_max, device)

    def compute(self, l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        Y = SphericalHarmonicsFunction.apply(l_max, self.prefactors, x, y, z)
        return Y


class SphericalHarmonicsFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, l_max: int, Flm, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

        r = torch.sqrt(x**2+y**2+z**2)
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        Phi = phi_dependent_recursions(l_max, x, y)
        Y = combine_into_spherical_harmonics(Flm, Qlm, Phi, r)

        ctx.Flm = Flm
        ctx.Qlm = Qlm
        ctx.Phi = Phi
        ctx.r = r
        ctx.save_for_backward(x, y, z)

        return tuple(Y)


    @staticmethod
    def backward(ctx, *grad_output):

        x, y, z = ctx.saved_tensors
        grad_Y = SphericalHarmonicsFunction.spherical_harmonics_custom_gradients(ctx.Flm, ctx.Qlm, ctx.Phi, x, y, z, ctx.r)
        d_input_d_x, d_input_d_y, d_input_d_z = SphericalHarmonicsFunction.calculate_gradients(grad_output, grad_Y)

        return None, None, d_input_d_x, d_input_d_y, d_input_d_z


    @torch.jit.script
    def spherical_harmonics_custom_gradients(Flm: List[torch.Tensor], Qlm: List[torch.Tensor], Phi: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, r: torch.Tensor):

        l_max = (Phi.shape[1] - 1) // 2

        # theta-dependent component of the spherical harmonics:
        Qlm = modified_associated_legendre_polynomials(l_max, z, r)
        grad_Qlm = modified_associated_legendre_polynomials_derivatives(Qlm, x, y)
        
        # phi-dependent component of the spherical harmonics:
        Phi = phi_dependent_recursions(l_max, x, y)
        grad_Phi = phi_dependent_recursions_derivatives(Phi)

        grad_Y = combine_into_spherical_harmonics_gradients(Flm, Qlm, Phi, grad_Qlm, grad_Phi, x, y, z, r)
        return grad_Y


    @torch.jit.script
    def calculate_gradients(grad_output: List[torch.Tensor], grad_Y: List[List[torch.Tensor]]):
        l_max = len(grad_output) - 1

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

        return d_input_d_x, d_input_d_y, d_input_d_z

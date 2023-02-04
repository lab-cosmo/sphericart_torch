import torch
from typing import List
from .theta import modified_associated_legendre_polynomials
from .phi import phi_dependent_recursions
from .combine import combine_into_spherical_harmonics


@torch.jit.script
def reference_spherical_harmonics(l_max: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

    r = torch.sqrt(x**2+y**2+z**2)
    Qlm = modified_associated_legendre_polynomials(l_max, z, r)
    Phi = phi_dependent_recursions(l_max, x, y)
    Y = combine_into_spherical_harmonics(Qlm, Phi, r)

    return Y

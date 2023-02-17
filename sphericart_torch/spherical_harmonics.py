import torch
from .prefactors import compute_prefactors
from spherical_harmonics_extension import spherical_harmonics


class SphericalHarmonics:

    def __init__(self, l_max, device):
        self.l_max = l_max
        self.prefactors = compute_prefactors(l_max, device)

    def compute(self, xyz):
        Y_tilde = spherical_harmonics(self.l_max, self.prefactors, xyz)

        r = torch.sqrt(torch.sum(xyz**2, dim = 1))
        r_l = torch.cat(
            [(r**(-l)).unsqueeze(dim=-1).repeat(1, 2*l+1) for l in range(self.l_max+1)],
            dim = 1
        )
        Y = Y_tilde*r_l
        return Y

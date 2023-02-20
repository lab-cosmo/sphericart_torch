import torch
from .prefactors import compute_prefactors
from spherical_harmonics_extension import spherical_harmonics


class SphericalHarmonics:

    def __init__(self, l_max, device, normalize=False):
        self.l_max = l_max
        self.prefactors = compute_prefactors(l_max, device)
        self.normalize = normalize

    def compute(self, xyz):
        if self.normalize:
            r = torch.sqrt(torch.sum(xyz**2, dim=1, keepdim=True))
            xyz = xyz / r
        Y = spherical_harmonics(self.l_max, self.prefactors, xyz.contiguous())
        return Y

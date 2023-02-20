import torch
import numpy as np
import scipy as sp
from scipy import special
import sphericart_torch
import e3nn

torch.set_default_dtype(torch.float64)
xyz = torch.rand(10, 3)

def test_sh_against_e3nn(xyz: torch.Tensor, l_max: int):

    # e3nn spherical harmonics:
    yzx = xyz[:, [1, 2, 0]]   # e3nn y, z, x ordering
    
    sh_e3nn = e3nn.o3.spherical_harmonics(range(l_max+1), yzx, normalize=True, normalization='integral')
    
    # sphericart spherical harmonics:
    sh_calculator = sphericart_torch.SphericalHarmonics(l_max, "cpu", normalize=True)
    sh_sphericart = sh_calculator.compute(xyz)
    
    assert torch.allclose(sh_sphericart, sh_e3nn)

l_max = 6
test_sh_against_e3nn(xyz, l_max)

print("Assertions passed successfully!")

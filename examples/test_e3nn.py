import torch
import numpy as np
import scipy as sp
from scipy import special
import torch_sh
import e3nn

torch.set_default_dtype(torch.float64)
xyz = torch.rand(10, 3)

def test_sh_against_e3nn(xyz: torch.Tensor, l_max: int):

    # e3nn spherical harmonics:
    #all_together = torch.stack([y, z, x], dim=1)
    
    sh_e3nn = e3nn.o3.spherical_harmonics(range(l_max+1), xyz, normalize=True, normalization='integral')

    print (sh_e3nn, sh_e3nn.shape)
    
    # sphericart spherical harmonics:
    sh_calculator = torch_sh.SphericalHarmonics(l_max, "cpu")
    sh_sphericart = sh_calculator.compute(xyz)

    print (sh_sphericart, sh_sphericart.shape)
    
    for l in range(l_max+1):
        print (sh_sphericart[l])
        assert torch.allclose(sh_sphericart[l], sh_e3nn[:, l**2:(l+1)**2])

l_max = 3
test_sh_against_e3nn(xyz, l_max)

print("Assertions passed successfully!")

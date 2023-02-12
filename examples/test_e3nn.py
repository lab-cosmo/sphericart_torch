import torch
import numpy as np
import scipy as sp
from scipy import special
import torch_sh
import e3nn

x = torch.rand(10)
y = torch.rand(10)
z = torch.rand(10)

def test_sh_against_e3nn(x: np.ndarray, y: np.ndarray, z: np.ndarray, l_max: int):

    # e3nn spherical harmonics:
    all_together = torch.stack([y, z, x], dim=1)
    sh_e3nn = e3nn.o3.spherical_harmonics(range(l_max+1), all_together, normalize=True, normalization='integral')

    # sphericart spherical harmonics:
    sh_calculator = torch_sh.SphericalHarmonics(l_max, "cpu")
    sh_sphericart = sh_calculator.compute(l_max, x, y, z)

    for l in range(l_max+1):
        assert torch.allclose(sh_sphericart[l], sh_e3nn[:, l**2:(l+1)**2])

l_max = 6
test_sh_against_e3nn(x, y, z, l_max)

print("Assertions passed successfully!")

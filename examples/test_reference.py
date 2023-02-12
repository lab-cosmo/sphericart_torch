import torch
import numpy as np
import scipy as sp
from scipy import special
import torch_sh
np.random.seed(0)

x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)

def test_sh_against_scipy(x: np.ndarray, y: np.ndarray, z: np.ndarray, l: int, m: int):

    # Scipy spherical harmonics:
    # Note: scipy's theta and phi are the opposite of those in our convention
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    complex_sh_scipy_l_m = sp.special.sph_harm(m, l, phi, theta)
    complex_sh_scipy_l_negm = sp.special.sph_harm(-m, l, phi, theta)

    if m > 0:
        sh_scipy_l_m = ((complex_sh_scipy_l_negm+(-1)**m*complex_sh_scipy_l_m)/np.sqrt(2.0)).real
    elif m < 0:
        sh_scipy_l_m = ((-complex_sh_scipy_l_m+(-1)**m*complex_sh_scipy_l_negm)/np.sqrt(2.0)).imag
    else: # m == 0
        sh_scipy_l_m = complex_sh_scipy_l_m.real

    # Torch spherical harmonics:
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)

    sh_calculator = torch_sh.ReferenceSphericalHarmonics(l, "cpu")
    sh_torch = sh_calculator.compute(l, x, y, z)
    sh_torch_l_m = sh_torch[l][:, l+m]

    assert torch.allclose(torch.tensor(sh_scipy_l_m), sh_torch_l_m), f"assertion failed for l={l}, m={m}"


l_max = 6
for l in range(0, l_max+1):
    for m in range(-l, l+1):
        test_sh_against_scipy(x, y, z, l, m)

print("Assertions passed successfully!")

import torch
import numpy as np
import scipy as sp
from scipy import special
import torch_sh
torch.set_default_dtype(torch.float64)

xyz = torch.rand(10, 3)

def test_sh_against_scipy(xyz, l, m):

    # Scipy spherical harmonics:
    # Note: scipy's theta and phi are the opposite of those in our convention
    xyz_np = xyz.numpy()
    x = xyz_np[:, 0]
    y = xyz_np[:, 1]
    z = xyz_np[:, 2]
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
    sh_calculator = torch_sh.SphericalHarmonics(l, "cpu")
    sh_torch = sh_calculator.compute(xyz)
    sh_torch_l_m = sh_torch[:, l**2+l+m]

    print(l, m)
    print(torch.tensor(sh_scipy_l_m))
    print(sh_torch_l_m)
    assert torch.allclose(torch.tensor(sh_scipy_l_m), sh_torch_l_m), f"assertion failed for l={l}, m={m}"


l_max = 8
for l in range(0, l_max+1):
    for m in range(-l, l+1):
        test_sh_against_scipy(xyz, l, m)

print("Assertions passed successfully!")

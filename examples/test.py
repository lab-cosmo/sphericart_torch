import torch
import numpy as np
import scipy as sp
import torch_sh
np.random.seed(0)

x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)

# Scipy spherical harmonics:
# Note: grrrrr scipy's theta and phi are the opposite of those in our convention
r = np.sqrt(x**2+y**2+z**2)
theta = np.arccos(z/r)
phi = np.arctan2(y, x)
complex_sh_scipy_5_2 = sp.special.sph_harm(2, 5, phi, theta)
complex_sh_scipy_5_n2 = sp.special.sph_harm(-2, 5, phi, theta)
sh_scipy_5_2 = ((complex_sh_scipy_5_n2+(-1)**2*complex_sh_scipy_5_2)/np.sqrt(2.0)).real

# Torch spherical harmonics:
x = torch.tensor(x)
y = torch.tensor(y)
z = torch.tensor(z)

sh_torch = torch_sh.spherical_harmonics(5, x, y, z)
sh_torch_5_2 = sh_torch[5][:, 5+2]

assert(torch.allclose(torch.tensor(sh_scipy_5_2), sh_torch_5_2))
print("Assertion passed successfully!")

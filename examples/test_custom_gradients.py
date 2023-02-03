import torch
import torch_sh

torch.set_default_dtype(torch.float64)
torch.random.manual_seed(0)

x = torch.rand((10,))
y = torch.rand((10,))
z = torch.rand((10,))


def check_gradients_against_finite_differences(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, l_max: int):

    # Finite difference gradients:
    delta = 1e-6
    
    xplus = x + delta
    xminus = x - delta
    yplus = y + delta
    yminus = y - delta
    zplus = z + delta
    zminus = z - delta

    sh_x_plus = torch_sh.spherical_harmonics(l_max, xplus, y, z)
    sh_x_minus = torch_sh.spherical_harmonics(l_max, xminus, y, z)
    sh_y_plus = torch_sh.spherical_harmonics(l_max, x, yplus, z)
    sh_y_minus = torch_sh.spherical_harmonics(l_max, x, yminus, z)
    sh_z_plus = torch_sh.spherical_harmonics(l_max, x, y, zplus)
    sh_z_minus = torch_sh.spherical_harmonics(l_max, x, y, zminus)

    xgrad = [(sh_x_plus[l] - sh_x_minus[l]) / (2.0 * delta) for l in range(l_max+1)]
    ygrad = [(sh_y_plus[l] - sh_y_minus[l]) / (2.0 * delta) for l in range(l_max+1)]
    zgrad = [(sh_z_plus[l] - sh_z_minus[l]) / (2.0 * delta) for l in range(l_max+1)]

    finite_difference_gradients = [xgrad, ygrad, zgrad]

    # Analytical gradients:
    analytical_gradients = torch_sh.spherical_harmonics_custom_gradients(l_max, x, y, z)

    # Assertions:
    for l in range(l_max+1):
        assert torch.allclose(finite_difference_gradients[0][l], analytical_gradients[0][l])
        assert torch.allclose(finite_difference_gradients[1][l], analytical_gradients[1][l])
        assert torch.allclose(finite_difference_gradients[2][l], analytical_gradients[2][l])


l_max = 6
check_gradients_against_finite_differences(x, y, z, l_max)
print("Assertions passed successfully!")

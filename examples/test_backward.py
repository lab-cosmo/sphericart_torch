import torch
import sphericart_torch

torch.set_default_dtype(torch.float64)
torch.random.manual_seed(0)
device = "cpu"

xyz = torch.rand((10, 3))


def check_gradients_against_finite_differences(xyz: torch.Tensor, l_max: int):

    sh_calculator = sphericart_torch.SphericalHarmonics(l_max, device)

    # Finite differences:
    delta = 1e-6
    xyz_x_plus = xyz.clone()
    xyz_x_minus = xyz.clone()
    xyz_y_plus = xyz.clone()
    xyz_y_minus = xyz.clone()
    xyz_z_plus = xyz.clone()
    xyz_z_minus = xyz.clone()
    xyz_x_plus[:, 0] += delta
    xyz_x_minus[:, 0] -= delta
    xyz_y_plus[:, 1] += delta
    xyz_y_minus[:, 1] -= delta
    xyz_z_plus[:, 2] += delta
    xyz_z_minus[:, 2] -= delta

    x_grad = (torch.sum(sh_calculator.compute(xyz_x_plus), dim=1)-torch.sum(sh_calculator.compute(xyz_x_minus), dim=1))/(2.0*delta)
    y_grad = (torch.sum(sh_calculator.compute(xyz_y_plus), dim=1)-torch.sum(sh_calculator.compute(xyz_y_minus), dim=1))/(2.0*delta)
    z_grad = (torch.sum(sh_calculator.compute(xyz_z_plus), dim=1)-torch.sum(sh_calculator.compute(xyz_z_minus), dim=1))/(2.0*delta)

    # Analytical gradients:
    xyz.requires_grad = True
    sh = sh_calculator.compute(xyz)
    dummy_loss = torch.sum(sh)
    dummy_loss.backward()

    # Assertions:
    assert torch.allclose(xyz.grad[:, 0], x_grad)
    assert torch.allclose(xyz.grad[:, 1], y_grad)
    assert torch.allclose(xyz.grad[:, 2], z_grad)


l_max = 6
check_gradients_against_finite_differences(xyz, l_max)
print("Assertions passed successfully!")

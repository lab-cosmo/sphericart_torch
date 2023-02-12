import torch
import torch_sh

torch.set_default_dtype(torch.float64)
torch.random.manual_seed(0)
device = "cpu"

x = torch.rand((10,))
y = torch.rand((10,))
z = torch.rand((10,))


def check_gradients_against_autograd(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, l_max: int):

    # Autograd gradients:
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    sh_calculator = torch_sh.ReferenceSphericalHarmonics(l_max, device)
    sh = sh_calculator.compute(l_max, x, y, z)
    dummy_loss = torch.zeros((1,), dtype = torch.get_default_dtype(), device = device)
    for tensor in sh:
        dummy_loss += torch.sum(tensor)
    dummy_loss.backward()
    x_grad = x.grad.clone()
    y_grad = y.grad.clone()
    z_grad = z.grad.clone()
    x.grad.zero_()
    y.grad.zero_()
    z.grad.zero_()

    # Custom gradients:
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    sh_calculator = torch_sh.SphericalHarmonics(l_max, device)
    sh = sh_calculator.compute(l_max, x, y, z)
    dummy_loss = torch.zeros((1,), dtype = torch.get_default_dtype(), device = device)
    for tensor in sh:
        dummy_loss += torch.sum(tensor)
    dummy_loss.backward()

    # Assertions:
    for l in range(l_max+1):
        assert torch.allclose(x.grad, x_grad)
        assert torch.allclose(y.grad, y_grad)
        assert torch.allclose(z.grad, z_grad)


l_max = 6
check_gradients_against_autograd(x, y, z, l_max)
print("Assertions passed successfully!")

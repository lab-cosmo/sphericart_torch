import torch
from typing import List


def reference_spherical_harmonics_gradients(sh_object: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):

    l_max = len(sh_object) - 1
    x_grad = []
    y_grad = []
    z_grad = []

    for l in range(l_max+1):
        x_grad_l = []
        y_grad_l = []
        z_grad_l = []

        for m in range(-l, l+1):

            x_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = x,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            x_grad_l.append(x_grad_l_m)

            y_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = y,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            y_grad_l.append(y_grad_l_m)

            z_grad_l_m = torch.autograd.grad(
                outputs = sh_object[l][:, l+m],
                inputs = z,
                grad_outputs = torch.ones_like(sh_object[l][:, l+m]),
                retain_graph = True
            )[0]
            z_grad_l.append(z_grad_l_m)

        x_grad_l = torch.stack(x_grad_l, dim=-1)
        y_grad_l = torch.stack(y_grad_l, dim=-1)
        z_grad_l = torch.stack(z_grad_l, dim=-1)

        x_grad.append(x_grad_l)
        y_grad.append(y_grad_l)
        z_grad.append(z_grad_l)

    gradients = [x_grad, y_grad, z_grad]
    return gradients

import torch
import torch_sh
import time
torch.set_default_dtype(torch.float64)
from torch.profiler import profile

device = "cpu"

x = torch.rand((500,), device=device, requires_grad=True)
y = torch.rand((500,), device=device, requires_grad=True)
z = torch.rand((500,), device=device, requires_grad=True)

l_max = 10
reference_calculator = torch_sh.ReferenceSphericalHarmonics(l_max, device)

print("Forward pass")
start_time = time.time()
sh = reference_calculator.compute(l_max, x, y, z)
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

dummy_loss = torch.zeros((1,), device=device)
for tensor in sh:
    dummy_loss += torch.sum(tensor)

print()
print("Backward pass")
start_time = time.time()
dummy_loss.backward()
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

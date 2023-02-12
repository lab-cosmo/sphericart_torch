import torch
import e3nn
import time
from torch.profiler import profile
torch.set_default_dtype(torch.float64)

device = "cpu"
l_max = 10

x = torch.rand((500,), device=device, requires_grad=True)
y = torch.rand((500,), device=device, requires_grad=True)
z = torch.rand((500,), device=device, requires_grad=True)
all_together = torch.stack([y, z, x], dim = 1)

print("Forward pass")
start_time = time.time()
sh = e3nn.o3.spherical_harmonics(range(l_max+1), all_together, normalize=True, normalization='integral')
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

dummy_loss = torch.sum(sh)

print()
print("Backward pass")
start_time = time.time()
dummy_loss.backward()
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

#print()
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

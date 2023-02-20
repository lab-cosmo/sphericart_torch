import torch
import e3nn
import time
from torch.profiler import profile
torch.set_default_dtype(torch.float64)

device = "cpu"
l_max = 10

#yzx = torch.rand((100000, 3), device=device, requires_grad=False)
#sh = e3nn.o3.spherical_harmonics(range(l_max+1), yzx, normalize=True, normalization='integral')

yzx = torch.rand((10000, 3), device=device, requires_grad=True)

print("Forward pass")
start_time = time.time()
sh = e3nn.o3.spherical_harmonics(range(l_max+1), yzx, normalize=True, normalization='integral')
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

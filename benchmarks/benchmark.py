import torch
import sphericart_torch
import time
torch.set_default_dtype(torch.float64)
from torch.profiler import profile

device = "cpu"

l_max = 10
sh_calculator = sphericart_torch.SphericalHarmonics(l_max, device)

xyz = torch.rand((1000000, 3), device=device, requires_grad=True)

print("Forward pass")
start_time = time.time()
if True: #with profile() as prof:
    sh = sh_calculator.compute(xyz)
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

dummy_loss = torch.sum(sh)

print()
print("Backward pass")
start_time = time.time()
if True: # with profile() as prof:
    dummy_loss.backward()
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

print()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

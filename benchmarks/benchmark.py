import torch
import torch_sh
import time
torch.set_default_dtype(torch.float64)

device = "cpu"
l_max = 8

nsph=10
x = torch.rand((nsph,), device=device, requires_grad=True)
y = torch.rand((nsph,), device=device, requires_grad=True)
z = torch.rand((nsph,), device=device, requires_grad=True)

#sh = torch_sh.spherical_harmonics(l_max, x, y, z)


nsph = 400000

x = torch.rand((nsph,), device=device, requires_grad=True)
y = torch.rand((nsph,), device=device, requires_grad=True)
z = torch.rand((nsph,), device=device, requires_grad=True)

print("Forward pass")
start_time = time.time()
sh = torch_sh.spherical_harmonics(l_max, x, y, z)
print(x[0],sh[2][:,0])
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")

ref_loss = torch.zeros((1,), device=device)
for tensor in sh:
    ref_loss += torch.sum(tensor)

print()
print("Backward pass")
start_time = time.time()
ref_loss.backward()
finish_time = time.time()
print(f"done in {finish_time-start_time} seconds")


sh = torch_sh.SphericalHarmonics(l_max)

#x = torch.rand((nsph,), device=device, requires_grad=True)
#y = torch.rand((nsph,), device=device, requires_grad=True)
#z = torch.rand((nsph,), device=device, requires_grad=True)

torch_time_fw = -time.time()
res = sh(x,y,z)
print(x[0],res[2][:,0])
torch_time_fw += time.time()
dummy_loss = torch.zeros((1,), device=device)
for tensor in res:
    dummy_loss += torch.sum(tensor)
torch_time_bw = -time.time()
dummy_loss.backward()
torch_time_bw += time.time()
sh.zero_grad()

script_sh = torch.jit.script(torch_sh.SphericalHarmonics(l_max))
script_time_fw = -time.time()
res = script_sh(x,y,z)
script_time_fw += time.time()
dummy_loss = torch.zeros((1,), device=device)
for tensor in res:
    dummy_loss += torch.sum(tensor)
script_time_bw = -time.time()
dummy_loss.backward()
script_time_bw += time.time()
print("chk: ", ref_loss, dummy_loss)

print("module time torch  (fw, bw)  ", torch_time_fw, torch_time_bw)
print("module time script (fw, bw)  ", script_time_fw, script_time_bw)

sh = torch_sh.SphericalHarmonics(l_max, forward_derivatives=True)
torch_time_fw = -time.time()
res = sh(x,y,z)
print(x[0],res[2][:,0])
torch_time_fw += time.time()
print("module time fwgrad  (fw, bw)  ", torch_time_fw)
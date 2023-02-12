import numpy as np
import torch


def compute_prefactors(l_max, device):
    Flm = torch.empty(((l_max+1)*(l_max+2)//2,), dtype = torch.get_default_dtype(), device = device)
    for l in range(l_max+1):
        Flm[l*(l+1)//2] = np.sqrt((2*l+1)/(2.0*np.pi))
        for m in range(1, l+1):
            Flm[l*(l+1)//2+m] = -Flm[l*(l+1)//2+m-1]/(np.sqrt((l+m)*(l+1-m)))
    return Flm

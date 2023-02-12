import numpy as np
import torch


def compute_prefactors(l_max, device):
    Flm = []
    for l in range(l_max+1):
        Flm.append(torch.empty((l+1,), dtype = torch.get_default_dtype(), device = device))
        Flm[l][0] = np.sqrt((2*l+1)/(2.0*np.pi))
        for m in range(1, l+1):
            Flm[l][m] = -Flm[l][m-1]/(np.sqrt((l+m)*(l+1-m)))

    return Flm

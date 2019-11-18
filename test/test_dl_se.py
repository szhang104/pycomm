import numpy as np
import scipy as sp
from simple import *
from scipy.io.matlab import loadmat

data = loadmat("DL_SE")

def test_dlse():
    CONFIG = {
        "cell": 16,
        "antenna_per_BS": 10, # no of BS antennas
        "user_per_cell": 10, # no of single-antenna user
        "bandwidth": 20e6,
        "kappa": 2, # path loss exponent
    }
    H = np.transpose(data["H"], (1,4,3,2,0))
    Hhat = np.transpose(data["Hhat"], (1,4,3,2,0))
    V = get_precoding(Hhat, method="ZF", local_cell_info=True)
    dlse = DL_SE(H, V, loop=False)
    dlse_t = data["SE_ZF_perfect"].transpose()
    return
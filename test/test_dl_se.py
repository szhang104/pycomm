import numpy as np
import scipy as sp
from simple import *
from scipy.io.matlab import loadmat

data = loadmat("DL_SE")
signal_ZF = np.transpose(data["signal_ZF"], (2, 1, 0))
intraInterf_ZF = data["intraInterf_ZF"].transpose()
interInterf_ZF = data["interInterf_ZF"].transpose()
dlse_t = data["SE_ZF_perfect"].transpose()
V_ZF = data["V_ZF"].transpose()
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
    dlse, sig, intra, inter = DL_SE(H, V, loop=False)
    assert np.allclose(V[499,15], V_ZF)
    assert np.allclose(sig, signal_ZF)
    assert np.allclose(intra, intraInterf_ZF)
    return
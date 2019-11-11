import numpy as np
from numpy import ndarray
import scipy as sp
# https://github.com/emilbjornson/optimal-beamforming/blob/master/simulationFigure3.m
from model_setup import channel_stat_setup

CONFIG = {
    "cell": 1,
    "antenna_per_BS": 100, # no of BS antennas
    "user_per_cell": 50, # no of single-antenna user
    "bandwidth": 20e6,
    "kappa": 2, # path loss exponent
}


def solve_left(A, B, a_is_hermitian=False):
    if a_is_hermitian:
        return np.linalg.solve(A, B.conj().transpose()).conj().transpose()
    else:
        return np.linalg.solve(A.conj().transpose(), B.conj().transpose(

        )).conj().transpose()


def zf_combining(H):
    """

    Parameters
    ----------
    H: just the cell of interest

    Returns
    -------

    """
    H1 = H
    A = H1.conj().transpose() @ H1 + 1e-12 * np.eye(H1.shape[1])
    B = H1
    res = solve_left(A, B, a_is_hermitian=True)
    return res



def get_H_rayleigh_unit(no_bs_ant,
                         no_user_per_cell,
                         no_cell,
                         no_realization=5):
    # Generate uncorrelated Rayleigh fading channel realizations with unit
    # variance
    randn2 = np.random.randn
    H = randn2(no_cell, no_cell, no_user_per_cell, no_bs_ant, no_realization) + \
        1j * randn2(no_cell, no_cell, no_user_per_cell, no_bs_ant, no_realization)
    return np.sqrt(0.5) * H



def get_channel_local_scatter(no_realization=10):
    # return shape:  no_real x no_cell x no_cell x no_user_per_cell x no_bs_ant
    R, gain_db = channel_stat_setup(CONFIG["cell"],
                               CONFIG["user_per_cell"],
                               CONFIG["antenna_per_BS"],
                               asd_degs=[30,], accuracy=2)
    # shape is no_bs_ant x no_bs_ant x no_user_cell x L x L x no_asd_degs

    R_gain = R[:, :, :, :, :, 0] * np.power(10, gain_db / 10.0)
    R_gain = np.ascontiguousarray(np.transpose(R_gain[:, :, :, :, :], (4,3,2,1,0)))
    # now the shape is no_cell x no_cell x no_user_cell x no_bs_ant x no_bs_ant
    # for each user, the channel between some BS to it, what is the spatial
    # correlation. Therefore in total there are so many numbers:
    # no_user_cell * no_cell * no_cell * no_bs_ant * no_bs_ant
    H = get_H_rayleigh_unit(
        CONFIG["antenna_per_BS"],
        CONFIG["user_per_cell"],
        CONFIG["cell"],
        no_realization=no_realization)
    H_gain = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            for k in range(H.shape[2]):
                H_gain[i,j,k,] = sp.linalg.sqrtm(R_gain[i,j,k]) @ H[i,j,k]
    res = np.ascontiguousarray(np.transpose(H_gain, (4, 0, 1, 2, 3)))
    return res



def decision(H):
    """

    Parameters
    ----------
    H
        Channel coefficients between M BS antennas and K single-antenna
        users. Complex array of shape ba x M x k. The first dimension is for
        different instantiation.

    Returns
    -------

    """
    return



if __name__ == "__main__":
    H = get_channel_local_scatter(no_realization=1000)
    print(H.shape)
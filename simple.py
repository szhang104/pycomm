import numpy as np
from numpy import ndarray
import scipy as sp
# https://github.com/emilbjornson/optimal-beamforming/blob/master/simulationFigure3.m
from model_setup import channel_stat_setup
from numba import jit, njit, prange
from timeit import default_timer as timer
CONFIG = {
    "cell": 1,
    "antenna_per_BS": 100, # no of BS antennas
    "user_per_cell": 50, # no of single-antenna user
    "bandwidth": 20e6,
    "kappa": 2, # path loss exponent
}

def hermitian(X):
    return X.conj().swapaxes(-1, -2)

def solve_left(A, B, a_is_hermitian=False):
    """
    Solve X A = B.
    transformed to equivalent problem A^H X^H = B^H
    Parameters
    ----------
    A
    B
    a_is_hermitian

    Returns
    -------

    """
    if a_is_hermitian:
        return hermitian(np.linalg.solve(A, hermitian(B)))
    else:
        return hermitian(np.linalg.solve(hermitian(A), hermitian(B)))


def zf_combining(H):
    """

    Parameters
    ----------
    H: CSI
    Returns
    -------

    """
    H1 = H
    A = hermitian(H1) @ H1 + 1e-12 * np.eye(H1.shape[1])
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
    for _idx in np.ndindex(*H.shape[0:3]):
        H_gain[_idx] = sp.linalg.sqrtm(R_gain[_idx]) @ H[_idx]
    res = np.ascontiguousarray(np.transpose(H_gain, (4, 0, 1, 2, 3)))
    return res


def DL_SE(channel, precoding, loop=True):
    """
    Calculate the ergodic DL spectral efficiency using Eq. 4.38 & 4.39 in
    Bjornsson book.

    Parameters
    ----------
    channel:
        H, the actual channel state information in complex numbers.
        Shape: no_real x no_cell x no_cell x no_user_per_cell x no_bs_antenna.

        Each element at (q, l_1, l_2, k, m) is the q-th realization of the
        channel from the m-th antenna from BS l_1 to the single-antenna user k
        in BS L_2.

    precoding:
        W. Downlink transmit precoding matrix, already normalized to have 1
        magnitude.
        Shape: no_real x no_cell x no_user_per_cell x no_bs_antenna,
        with similar meaning as H.

    Returns
    -------
    SE:
        the downlink spectral density of all users, in bit/s/Hz.
        Shape: no_real x no_cell x no_user_per_cell.
    SINR:
        the signal-to-interference-and-noise ratio of all users. NOT in dB.
        Shape: no_real x no_cell x no_user_per_cell.
    """
    H, W = channel, precoding
    no_real, L, K, M = H.shape[0], H.shape[
        1], H[3], H[4]
    intercell_intf = np.zeros((L, K), dtype=np.complex)
    intracell_intf = np.zeros((no_real, L, K), dtype=np.complex)
    sig = np.zeros((no_real, L, K), dtype=np.complex)
    if loop:
        for n in range(no_real):
            for l in range(L):
                H_l2all = H[n, l] # L x K x M, the CSI between BS l and all
                # users
                for k in range(K):
                    # for all users in cell l
                    w = W[n, l, k] # M x 1
                    sig[n, l, k] = (np.abs(w.conj().T @ H[n, l, l, k])) ** 2
                    intracell_intf = H_l2all[l].conj() @ w # [K,]
                    inner_all = H_l2all.conj() @ w # [L, K]
                    intercell_intf = (np.abs(inner_all)) ** 2
                    intercell_intf[l] -= (np.abs(intracell_intf)) ** 2
                    intracell_intf = (np.abs(intracell_intf)) ** 2
                    intracell_intf[n, l, k] -= sig[n, l, k]
    else:
        pass


    return


def get_precoding(H, method="ZF", local_cell_info=True):
    if method == "ZF":
        algo = zf_combining
    if local_cell_info:
        no_cells = H.shape[1]
        for j in range(no_cells):
            W = algo(H[:,j,j])


    if method == "ZF":




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
    W = zf_combining(H)
    DL_SE(H, W)



import numpy as np
from numpy import ndarray
import scipy as sp
# https://github.com/emilbjornson/optimal-beamforming/blob/master/simulationFigure3.m
from model_setup import channel_stat_setup
from utils import hermitian, mldivide


CONFIG = {
    "cell": 4,
    "antenna_per_BS": 100, # no of BS antennas
    "user_per_cell": 50, # no of single-antenna user
    "bandwidth": 20e6,
    "kappa": 2, # path loss exponent
    "p_t_dl": 100, # downlink transmit power in mW
    "noise_figure": 7,
}

def noise_dbm():
    return -174 + 10 * np.log10(CONFIG["bandwidth"]) + CONFIG["noise_figure"]


def zf_combining(H):
    """

    Parameters
    ----------
    H: CSI of a single cell. no_real x no_user_per_cell x no_antenna
    Returns
    -------
    """
    H1 = H
    A = hermitian(H1) @ H1 + 1e-12 * np.eye(H1.shape[-1])
    B = H1
    res = mldivide(A, B, A_is_hermitian=True)
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
    if CONFIG["cell"] > 1 and CONFIG["cell"] < 4:
        no_BS_per_dim = np.array([1, CONFIG["cell"]])
    else:
        no_BS_per_dim = None
    R, gain_db = channel_stat_setup(CONFIG["cell"],
                               CONFIG["user_per_cell"],
                               CONFIG["antenna_per_BS"],
                            no_BS_per_dim=no_BS_per_dim,
                               asd_degs=[30,], accuracy=2)
    gain_db -= noise_dbm()
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


def DL_SE(channel, precoding, power=100, loop=True):
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
        V. Downlink transmit precoding matrix, already normalized to have 1
        magnitude.
        Shape: no_real x no_cell x no_user_per_cell x no_bs_antenna,
        with similar meaning as H.

    power:
        scalar. the power used when transmitting, in mW.

    Returns
    -------
    SE:
        the downlink spectral density of all users, in bit/s/Hz.
        Shape: no_real x no_cell x no_user_per_cell.
    SINR:
        the signal-to-interference-and-noise ratio of all users. NOT in dB.
        Shape: no_real x no_cell x no_user_per_cell.
    """
    H, V = channel, precoding
    W = complex_normalize(V, -1)

    no_real, L, K, M = H.shape[0], H.shape[
        1], H.shape[3], H.shape[4]
    intercell_intf = np.zeros((L, K))
    intracell_intf = np.zeros((no_real, L, K))
    sig = np.zeros((no_real, L, K))
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
        for n in range(no_real):
            for l in range(L):
                H_l = H[n, l] # (L, K, M)
                for k in range(K):
                    w_l = W[n, l] # (K, M)
                    H_llk = H_l[l, k] # (M, ) the channel b/w l-th BS to user k
                    p_l = np.abs(np.dot(w_l.conj(), H_llk)) ** 2
                    sig[n, l, k] = p_l[k]
                    intracell_intf[n, l, k] = p_l.sum() - p_l[k]
                    if L > 1:
                        idx_othercell = list(range(L))
                        idx_othercell.remove(l)
                        H_intercell = H[n, idx_othercell, l:l+1, k] # (L-1, 1, M) CSI,
                        # other cells
                        # to
                        # this user k
                        w_intercell = W[n, idx_othercell] #(L-1, K, M) other cell's precoding vec
                        p_inter = np.abs(w_intercell @ (H_intercell.swapaxes(
                            -1, -2))) ** 2
                        intercell_intf[l,k] += p_inter.sum() / no_real
                        # assert np.allclose(p_sig, np.abs(w_l[k].conj() @ H_llk)
                        #                    ** 2)
        int_noise = power * intercell_intf + power * intracell_intf + 1
        sinr = (power * sig / int_noise)
        dl_se = np.log2(1+sinr).mean(axis=0)

    return dl_se, sig, intracell_intf, intercell_intf


def get_precoding(H, method="ZF", local_cell_info=True):
    res = []
    if method == "ZF":
        algo = zf_combining
    if local_cell_info:
        no_cells = H.shape[1]
        for j in range(no_cells):
            res.append(algo(H[:,j,j]))
    return np.stack(res, axis=1)



def complex_normalize(X, axis=-1):
    """
    Normalize the complex n-dim array on the dimension axis
    Parameters
    ----------
    X: n-dimension complex array
    Returns
    -------
    """
    mags = np.linalg.norm(np.abs(X), axis=axis, keepdims=True)
    return X / mags

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
    H = get_channel_local_scatter(no_realization=10)
    W = get_precoding(H, method="ZF", local_cell_info=True)
    DL_SE(H, W, loop=False)



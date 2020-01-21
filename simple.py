import numpy as np
from numpy import ndarray
import scipy as sp
# https://github.com/emilbjornson/optimal-beamforming/blob/master/simulationFigure3.m
from model_setup import channel_stat_setup
from utils import hermitian, mldivide
import jax
from utils import mkl_matmul
import mkl


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
        # TODO: use Cholesky factorization to replace the slow matrix square
        # root operation.  However, it requires the matrix to be positive
        # semidefinite, which should be the case but due to the numerical error
        # is not always the case.
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


def SE(H, W):
    """
    A simplified implementation of spectral efficiency
    Parameters
    ----------
    H:
        the channel state, power is normalized w.r.t to noise power.
         (no_real, N, N, K, M)
    W:
        the precoding unnormalized, (no_ral, N, N, K, M)

    Returns
    -------
    the spectral efficiency per user. Normalized w.r.t to the bandwidth
        (no_real, N, K)
    """

    no_real, N, N, K, M = H.shape
    all_powers = np.swapaxes(np.swapaxes(H, 0, 1) @ hermitian(W), 0, 1)
    all_powers = np.abs(all_powers) ** 2



    # (no_real, N, N, K, K)
    # (no_real, n_t, n, k, k_neighbor)
    # the power coming from BS n_t to User k in BS n, using the
    # precoding of BS n_t to user k_neighbor in BS n1


    p_sig = np.zeros((no_real, N, K))
    p_int = np.zeros((no_real, N, K, N))
    sinr = np.zeros_like(p_sig)


    for r in range(no_real):
        for n in range(N):
            for k in range(K):
                p_sig[r, n, k] = all_powers[r, n, n, k, k]
                for n_t in range(N):
                    p_int[r, n, k, n_t] = all_powers[r, n_t, n, k].sum()
                    if n_t == n:
                        p_int[r, n, k, n_t] -= p_sig[r,n,k]
    sinr = p_sig / ((p_int).sum(axis=-1) + 1)
    return np.log2(1 + sinr), p_sig, p_int





def get_BS_power(precode, antenna_sel):
    """
    Calculate the BS power based on two parts:
    1. transmission activity, E(ww^H)
    2. each antenna costs a fixed amount of power for circuits/power amps

    Parameters
    ----------
    precode:
        precoding vectors, (_, N, K, M)
    antenna_sel
        antenna selection, zero-one (_, N, M)

    Returns
    -------
        per BS powers (_, N)

    """
    p_1 = np.sum(antenna_sel > 0, axis=2) * CONFIG["p_antenna"]
    p_0 = (np.abs(precode) ** 2).sum(axis=3).sum(axis=2)
    return p_0 + p_1 + CONFIG["p_fixed"] + CONFIG["p_lo"]


def solve_ee_greedy(H, ant_sel=True):
    """

    Parameters
    ----------
    H:
        the channel state, with large scale gain.
        (_, N, N, K, M)



    Returns
    -------
    precode:
        precoding vectors, with gain
        (_, N, N, K, M)

    antenna_sel:
        antenna selection vector, (no_real, N, M)
    """
    no_real, N, N, K, M = H.shape
    if ant_sel:

        antenna_sel = np.zeros((no_real, N, M), dtype=np.bool)

        # strongest K_0 antennas
        K_0 = int(M * 0.8)
        for r in range(no_real):
            for n in range(N):
                channel_power_ant = (np.abs(H[r, n, n]) ** 2).sum(axis=-2) # (M, )
                top_k = np.argsort(channel_power_ant)[0:K_0]
                antenna_sel[r, n][top_k] = True

        # or randomly

        H_n = (H.transpose(2, 3, 0, 1, 4) * antenna_sel).transpose(2, 3, 0,
                                                                   1, 4)
    else:
        antenna_sel = np.ones((no_real, N, M), dtype=np.bool)
        H_n = H
    W = get_precoding(H_n, method="ZF", local_cell_info=True)
    # power allocation
    W = np.sqrt(CONFIG["p_t_dl"]) * complex_normalize(W)
    return W, antenna_sel





CONFIG = {
    "cell": 16,
    "antenna_per_BS": 50, # no of BS antennas
    "user_per_cell": 10, # no of single-antenna user
    "bandwidth": 20e6,
    "noise_figure": 7,
    "p_t_dl": 0.2, # downlink transmit power in Watts PER UE
    "p_antenna": 0.4, # 0.2-0.4, per BS antenna power cost
    "p_fixed": 10.0,
    "p_lo": 0.2,
    "no_real": 30,
}

if __name__ == "__main__":
    # need to add BS location and UE location figure and config
    H = get_channel_local_scatter(no_realization=CONFIG["no_real"]) # (_, N, N,
    # K, M)
    W, antenna_sel = solve_ee_greedy(H, ant_sel=True) # (_, N, K, M)
    SE_users, sig_users, int_users = SE(H, W) # (_, N, K)
    SE_cells = SE_users.sum(axis=-1) # (_, N)
    power_cells = get_BS_power(W, antenna_sel) # (_, N)
    print("sel", SE_users.mean(), power_cells.mean(), np.mean(SE_cells.sum(
        axis=1)/power_cells.sum(axis=1)))

    W, antenna_sel = solve_ee_greedy(H, ant_sel=False) # (_, N, K, M)
    SE_users, sig_users, int_users = SE(H, W) # (_, N, K)
    SE_cells = SE_users.sum(axis=-1) # (_, N)
    power_cells = get_BS_power(W, antenna_sel) # (_, N)
    print("no_sel", SE_users.mean(), power_cells.mean(), np.mean(SE_cells.sum(
        axis=1)/power_cells.sum(axis=1)))




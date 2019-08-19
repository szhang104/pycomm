import numpy as np
from numpy import ndarray
from utils import running_avg

def mr_combining(H, ue):
    return H[:, ue]


def rzf_combining(H, ue, p):
    H1 = H[:, ue]
    A = p * (H1.conj().transpose() @ H1) + np.eye(H1.shape[1])
    B = p * H1
    return np.linalg.solve(A, B.conj().transpose()).conj().transpose()


def zf_combining(H: ndarray, ue):
    """

    Parameters
    ----------
    H: just the cell of interest

    Returns
    -------

    """
    H1 = H[:, ue]
    A = H1.conj().transpose() @ H1 + 1e-12 * np.eye(H1.shape[1])
    B = H1
    return np.linalg.solve(A, B.conj().transpose()).conj().transpose()


def smmse_combining(H: ndarray, ue, p, C):
    H1 = H[:, ue]
    A = p * (p * H1 @ H1.conj().transpose() + C + np.eye(H1.shape[0]))
    B = H1
    return np.linalg.solve(A, B)


def mmmse_combining(H: ndarray, ue, p, C):
    """
    returns the receive combining vector for multi-cell minimum-mean-square-error scheme
    Parameters
    ----------
    H: (M, K*L)
        the channel realization (or the estimations thereof) for all users in all cells
        and its own cell
    p:
        transmit power
    C:
        correlation of the error
    ue:
        a list of UE numbers in the cell of interest

    Returns
    -------

    """
    B = H[:, ue]
    A = p * (p * H @ H.conj().transpose() + C + np.eye(H.shape[0]))
    return np.linalg.solve(A, B)





def SE_UL(Hhat:ndarray, C:ndarray, R, tau_c, tau_p, realization_cnt, M, K, L, p):
    """
    calculates the uplink spectral efficiency for different receive combining
    Parameters
    ----------
    Hhat (M, realization_cnt, K, L, L)
        MMSE channel estimates
    C (M, M, K, L, L)
        estimation error correlation matrix with MMSE estimation
    R
    tau_c
    tau_p
    realization_cnt
    M
    K
    L
    p: float
        uplink transmit power (same for all here)

    Returns
    -------

    """
    methods = ['MR', 'RZF', 'MMMSE', 'ZF', 'SMMSE']
    V = {}
    # sum of all estimation error correlation matrices at every BS
    # shape (M, M, L)
    C_totM = np.reshape(p * C.sum(axis=(2, 3)), (M, M, L))
    # sum of intra-cell estimation error correlation matrices at every BS
    CR_totS = np.zeros([M, M, L], dtype=np.complex)
    for j in range(L):
        all_other_cells = np.ones((L,), dtype=np.bool)
        all_other_cells[j] = False
        CR_totS[:, :, j] = p * (C[:, :, :, j, j].sum(axis=2) + R[:, :, :, all_other_cells, j].sum(axis=(2,3)))

    prelog_factor = (tau_c - tau_p) / tau_c
    SE = {}
    for method in methods:
        SE[method] = np.zeros([K, L])

    for n in range(realization_cnt):
        for j in range(L):
            # matlab uses F order, shape(M, KL)
            Hhat_allj = Hhat[:, n, :, :, j].reshape(M, K*L, order='F')
            ue = np.arange(K*j, K*j + K)
            Hhat_j = Hhat_allj[:, K*j: K*j+K]
            V['MR'] = mr_combining(Hhat_allj, ue)
            V['RZF'] = rzf_combining(Hhat_allj, ue, p)
            V['ZF'] = zf_combining(Hhat_allj, ue)
            V['MMMSE'] = mmmse_combining(Hhat_allj, ue, p, C_totM[:, :, j])
            V['SMMSE'] = smmse_combining(Hhat_allj, ue, p, CR_totS[:, :, j])

            for k in range(K):
                for method in methods:
                    v = V[method][:, k]
                    # v: (M, ), Hhat: (M,)
                    numerator = p * (np.abs(v.conj() @ Hhat[:, n, k, j, j]) ** 2)
                    # Hhat_allj: (M, K*L)
                    denominator = p * np.sum(np.abs(v.conj() @ Hhat_allj) ** 2) + \
                                  v.conj() @ (C_totM[:, :, j] + np.eye(M)) @ v - numerator
                    SE[method][k, j] = running_avg(n, SE[method][k,j], prelog_factor * np.log2(1 + numerator / denominator).real)
    return SE
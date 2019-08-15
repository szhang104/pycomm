import numpy as np
from numpy import ndarray


def mr_combining(H, ue):
    return H[:, ue]


def rzf_combining(H, ue, p):
    H1 = H[:, ue]
    A = p * (H1.conj().transpose() @ H1) + np.eye(H1.shape[0])
    B = p * H1
    return np.linalg.solve(A.conj().transpose(), B.conj().transpose())


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
    return np.linalg.solve(A, B.conj().transpose())


def smmse_combining(H: ndarray, ue, p, C):
    H1 = H[:, ue]
    A = p * (p * H1.conj().tranpose() @ H1 + C[:, :, j] + np.eye(H1.shape[0]))
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
    A = p * (p * H @ H.conj().transpose() + C[:, :, j] + np.eye(H.shape[0]))
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
    # sum of all estimation error correlation matrices at every BS
    C_totM = np.reshape(p * C.sum(axis=2).sum(axis=3), (M, M, L))
    # sum of intra-cell estimation error correlation matrices at every BS
    for j in range(L):
        all_other_cells = np.ones((L,), dtype=np.bool)
        all_other_cells[j] = False
        CR_totS = C[:, :, :, j, j].sum(axis=2) + R[:, :, :, all_other_cells, j].sum(axis=2).sum(axis=3)

    for n in range(realization_cnt):
        for j in range(L):
            Hhat_allj = Hhat[:, n, :, :, j].reshape(M, K*L)
            ue = np.arange(K*j, K*j + K)
            Hhat_j = Hhat_allj[:, K*j: K*j+K]
            V_MR = mr_combining(Hhat_allj, ue)
            V_RZF = rzf_combining(Hhat_allj, ue, p)
            V_MMMSE = mmmse_combining(Hhat_allj, ue, p, C_totM)
            V_ZF = zf_combining(Hhat_allj, ue)
            V_SMMSE = smmse_combining(Hhat_allj, ue, CR_totS)

            for k in range(K):
                v = V[:, k]
                numerator = p * np.abs(v.conj().transpose() @ Hhat[:, n, k, j, j]) ** 2
                denomintor = p * np.sum(np.(v.conj().transpose() * Hhat_allj))??
                SE[k, j] += prelog_factor * np.log2(1 + numerator / denomintor).real

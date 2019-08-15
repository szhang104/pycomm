import numpy as np
import scipy as sp
from utils import randn2


def channel_estimates(R, channel_gain_db, realization_cnt, M, K, L, p, f):
    """returns channel realizations and their estimates
    Parameters
    ----------
    R: (M, M, K, L, L)
        the spatial correlation matrix for all UEs in the network.
        R[:, :, k, j, l] is the UE k in cell j and the BS in cell l
    channel_gain_db: (K, L, L)
        average channel gains in dB.
    realization_cnt: int
        number of channel realizations
    M:
        number of antennas per BS
    K:
        number of UEs in a cell
    L: int
        number of BSs(cells). for now only 16 works.
    p:
        uplink transmit power per UE (same here)
    f:
        pilot reuse factor

    Returns
    -------
    Hhat: (M, realization_cnt, K, L, L)
        estimates of the channel
    H: (M, realization_cnt, K, L, L)
        the real channel
    R: scaled input spatial correlation by the channel gains

    C: (M, M, K, L, L)
        estimation error correlation
    """

    # Generate uncorrelated Rayleigh fading channel realizations
    H = randn2(M, realization_cnt, K, L, L) + 1j * randn2(M, realization_cnt, K, L, L)

    # Prepare a matrix to save the channel gains per UE
    betas = np.zeros((K,L,L))

    # Go through all channels and apply the channel gains to the spatial
    # correlation matrices
    for j in range(L):
        for l in range(L):
            for k in range(K):
                if channel_gain_db[k, j, l] > -np.inf:
                    # Extract channel gain in linear scale
                    betas[k,j,l] = 10 ** (channel_gain_db[k, j, l] / 10)
                    # Apply channel gain to correlation matrix
                    R[:, :, k, j, l] = betas[k, j, l] * R[:, :, k, j, l]
                    # Apply correlation to the uncorrelated channel realizations
                    Rsqrt = sp.linalg.sqrtm(R[:, :, k, j, l])
                    H[:, :, k, j, l] = np.sqrt(0.5) * Rsqrt @ H[:, :, k, j, l]
                else:
                    betas[k,j,l] = 0
                    R[:, :, k, j, l] = 0
                    H[:, :, k, j, l] = 0

    # do the channel estimation
    tau_p = f * K
    # pilot reuse patterns. only work when there are 16 BS
    if f == 1:
        pilot_pattern = np.zeros((L,))
    elif f == 2:
        pilot_pattern = np.kron(np.ones((2,)), np.array([1,2,1,2,2,1,2,1]))
    elif f == 4:
        pilot_pattern = np.kron(np.ones((2,)), np.array([0,1,0,1,2,3,2,3]))
    elif f == 16:
        pilot_pattern = np.arange(L)
    else:
        raise NotImplementedError('Unknown f')

    # realizations of normalized noise
    Np = np.sqrt(0.5) * (
        randn2(M, realization_cnt, K, L, f) + 1j * randn2(M, realization_cnt, K, L, f)
    )

    # MMSE channel estimates and error correlation matrix
    Hhat_MMSE = np.zeros((M, realization_cnt, K, L, L), dtype=np.complex128)
    C_MMSE = np.zeros((M, M, K, L, L), dtype=np.complex128)

    for j in range(L):
        for g in range(f):
            # id's of the cells that have the same pilot group
            group_members = np.nonzero(g == pilot_pattern)[0]
            yp = np.sqrt(p) * tau_p * np.sum(H[:, :, :, group_members, j], axis=3) \
                + np.sqrt(tau_p) * Np[:, :, :, j, g]
            for k in range(K):
                PsiInv = p * tau_p * np.sum(R[:, :, k, group_members, j], axis=2) + np.eye(M)
                for l in group_members:
                    RPsi = np.linalg.solve(PsiInv.conjugate().transpose(), R[:, :, k, l, j]).conjugate().transpose()
                    Hhat_MMSE[:, :, k, l, j] = np.sqrt(p) * RPsi @ yp[:, :, k]
                    C_MMSE[:, :, k, l, j] = R[:, :, k, l, j] - p * tau_p * RPsi @ R[:, :, k, l, j]
    return Hhat_MMSE, C_MMSE, tau_p, R



import numpy as np
import scipy as sp
from scipy.linalg import solve
from pandas import DataFrame
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from numba.extending import overload
from numba import types
import time
from scipy.io.matlab import loadmat
# from naginterfaces.library import det, linsys, lapacklin

Kmax = 20
K = np.arange(1, Kmax+1)
C = [1, 2, 4, 8] # the M/K ratios
Mmax = Kmax * max(C)
SNR = 1
# strength of inter-cell interference
betabar = 1e-1

test = True

# Select number of Monte Carlo realizations for the line-of-sight (LoS)
#     angles and of the non-line-of-sight (NLoS) Rayleigh fading
numberOfRealizations = 500


# Generate NLoS channels using uncorrelated Rayleigh fading

@jit
def uncorrelated_rayleigh_channel(in_cnt, out_cnt, realization_cnt=1, realization_index_first=True):
    res = np.sqrt(0.5) * np.random.randn(in_cnt, out_cnt, realization_cnt) + 1j * np.random.randn(in_cnt, out_cnt, realization_cnt)
    if realization_index_first:
        res = np.transpose(res, axes=(2, 0, 1))
    return res

if test:
    data_matlab = loadmat('test/test_1_18.mat')
    H_NLoS_desired = data_matlab['H_NLoS_desired'].transpose(2,0,1).copy()
    H_NLoS_interfering = data_matlab['H_NLoS_interfering'].transpose(2,0,1).copy()
    SE_MMSE_NLoS_montecarlo_ref = data_matlab['SE_MMSE_NLoS_montecarlo']
    SE_MMSE_NLoS_nonlinear_ref = data_matlab['SE_MMSE_NLoS_nonlinear']
else:
    H_NLoS_desired = uncorrelated_rayleigh_channel(Mmax, Kmax, numberOfRealizations)
    H_NLoS_interfering = np.sqrt(betabar) * uncorrelated_rayleigh_channel(Mmax, Kmax, numberOfRealizations)


# %Preallocate matrices for storing the simulation results
SE_MMSE_NLoS_montecarlo = np.zeros((len(K), len(C)))
SE_MMSE_NLoS_nonlinear = np.zeros((len(K), len(C)))

# @jit
# def log2_det_nag(x):
#     fac_x = lapacklin.zgetrf(x)
#     res = det.complex_gen(fac_x.a, fac_x.ipiv)
#     return np.log2(res.d.real) + res.dexp[0]
#
# @jit
# def solve_herm_nag(a, b):
#     fac = lapacklin.zpotrf('U', a)
#     res = lapacklin.zpotrs('U', fac, b)
#     return res

@jit
def log2_det_hermitian(x):
    res = np.linalg.slogdet(x)
    return res[0].real * res[1] / np.log(2)

@jit
def SIC_SE(SNR, H_des, H_intf):
    M = H_des.shape[0]
    a = np.eye(M) + SNR * (H_des @ H_des.conj().T) + SNR * (H_intf @ H_intf.conj().T)
    b = np.eye(M) + SNR * (H_intf @ H_intf.conj().T)
    # return np.real(log2_det_nag(a) - log2_det_nag(b))
    return log2_det_hermitian(a) - log2_det_hermitian(b)

@overload(solve)
def mysolve(a, b):
    if isinstance(a, types.Array) and isinstance(b, types.Array):
        res = solve(a, b)
        def xxx(a, b):
            return res
        return xxx

@jit
def get_MMMSE_filter(SNR, H_des, H_intf):
    """
    return the filter coefficients of a Multicell-minimum square error filter.

    See eq.(1.42) in massive mimo book
    :param SNR:
    :param H_des:
    :param H_intf:
    :return:
    """
    M = H_des.shape[0]
    a = SNR * H_des @ H_des.conj().T
    b = SNR * H_intf @ H_intf.conj().T
    c = solve(a + b + np.eye(M), SNR * H_des)
    # c = solve_herm_nag(a + b + np.eye(M), SNR * H_des)
    return c

# Go through all Monte Carlo realizations
start = time.time()

for n in range(0, numberOfRealizations):
    # Output simulation progress
    print('{} realizations out of {}'.format(n, numberOfRealizations))

    # Go through the range of number of UEs
    for kindex, k in enumerate(K):
        # %Go through the range of antenna-UE ratios
        for cindex, c in enumerate(C):
            # %Compute the number of antennas
            M = k * c
            H_0 = np.ascontiguousarray(H_NLoS_desired[n, 0:M, 0:k])
            H_1 = np.ascontiguousarray(H_NLoS_interfering[n, 0:M, 0:k])

            # %Compute the SE with non-linear processing under NLoS propagation
            # %for one realization of the Rayleigh fading. We use the classic
            # %log-det formula for the uplink sum SE, when treating the
            # %inter-cell interference as colored noise
            SE_MMSE_NLoS_nonlinear[kindex,cindex] += SIC_SE(SNR, H_0, H_1) / numberOfRealizations
            # %realization of the Rayleigh fading
            # %Compute the M-MMSE combining vectors
            MMMSEfilter = get_MMMSE_filter(SNR, H_0, H_1)
            # %Compute the intra-cell channel powers after M-MMSE combining
            channelgainsIntracell = np.square(np.abs(MMMSEfilter.conj().T @ H_0))
            #
            # Extract the desired signal power for each UE
            signalpowers = np.diag(channelgainsIntracell)
            #
            # %Extract and compute interference powers for each UE
            interferencepowers = np.sum(channelgainsIntracell, axis=1) - signalpowers + \
                                 np.sum(np.square(np.abs(MMMSEfilter.conj().T @ H_1)), axis=1)
            #
            # %Compute the effective 1/SNR after noise amplification
            scalednoisepower = (1/SNR) * np.sum(np.square(np.abs(MMMSEfilter.conj().T)), axis=1)
            #
            # %Compute the uplink SE with M-MMSE combining
            SE_MMSE_NLoS_montecarlo[kindex,cindex] += np.sum(np.log2(1 + signalpowers / (interferencepowers + scalednoisepower)))/numberOfRealizations
print(time.time() - start)

if test:
    print(np.linalg.norm(SE_MMSE_NLoS_montecarlo_ref - SE_MMSE_NLoS_montecarlo),
          np.linalg.norm(SE_MMSE_NLoS_nonlinear_ref - SE_MMSE_NLoS_nonlinear))

fig = plt.figure(1)
line_styles = ['r--', 'k-', 'b--', 'k:']
line_styles.reverse()
for cindex, c in enumerate(C):
    plt.plot(K, SE_MMSE_NLoS_montecarlo[:, cindex] / SE_MMSE_NLoS_nonlinear[:, cindex], line_styles[cindex], label='M/K={}'.format(c), linewidth=1)

plt.xlabel('Number of UEs (K)')
plt.ylabel('Fraction of non-linear performance')

plt.legend()
plt.ylim([0.5, 1])
plt.show()

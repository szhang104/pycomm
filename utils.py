import numpy as np
from numpy import ndarray
from scipy.special import erfinv
import ctypes

mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")

def hermitian(X):
    return X.conj().swapaxes(-1, -2)

def randn2(*args,**kwargs):
    '''
    Calls rand and applies inverse transform sampling to the output.
    Since Matlab is column-major and numpy is row-major by default, so the random number generation
    first gets a reversed version instead
    '''
    args_r = tuple(reversed(args))
    uniform = np.random.rand(*args_r)
    uniform = uniform.transpose()
    return np.sqrt(2) * erfinv(2 * uniform - 1)


def noise_variance_db(B, noise_fig=7):
    return -174 + 10 * np.log10(B) + noise_fig


def running_avg(i, old, new):
    return (old * i + new) / (i + 1.0)


def mldivide(A: ndarray, B: ndarray, A_is_hermitian=False):
    if A_is_hermitian:
        return hermitian(np.linalg.solve(A, hermitian(B)))
    else:
        return hermitian(np.linalg.solve(hermitian(A), hermitian(B)))


def rayleigh_channel(t_cnt, r_cnt, t_antenna, r_antenna_cnt, squeeze=True, realization_cnt=1, gaussian_rng=None):
    if gaussian_rng is None:
        gaussian_rng = np.random.randn
    elif gaussian_rng == 'debug':
        gaussian_rng = randn2
    res:ndarray = gaussian_rng(realization_cnt, t_cnt, r_cnt, t_antenna, r_antenna_cnt) + \
          1j * gaussian_rng(realization_cnt, t_cnt, r_cnt, t_antenna, r_antenna_cnt)
    if squeeze:
        res = res.squeeze()
    res = np.ascontiguousarray(res, dtype=np.float)
    return res


def correlated_rayleigh(ue_cnt, bs_cnt, antenna_cnt, R, squeeze=True, realization_cnt=1, gaussian_rng=None):
    # R is the correlation matrices and hence Hermitian
    # R shape (..., antenna_cnt, antenna_cnt)
    # w shape (..., antenna_cnt), v shape(..., antenna_cnt, antenna_cnt)
    # return shape (..., M)

    # implements eq. 2.4, h = R^0.5 e_inversehat = U D^0.5 U^H e_inversehat ~ U D^0.5 e, where e ~ N_C(0_r, I_r), r
    # is the number of non-zero eigenvalues of R
    if gaussian_rng is None:
        gaussian_rng = np.random.randn
    elif gaussian_rng == 'debug':
        gaussian_rng = randn2
    w, v = np.linalg.eigh(R)
    uncorrelated = gaussian_rng(realization_cnt, ue_cnt, bs_cnt, antenna_cnt) + 1j * gaussian_rng(realization_cnt, ue_cnt, bs_cnt, antenna_cnt)
    h = v @ np.expand_dims(np.sqrt(w) * uncorrelated, axis=len(uncorrelated.shape))
    return h


def mkl_matmul(A, B, A_trans=None, B_trans=None):
    # def trans_code(x):
    #     # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    #     if x is None:
    #         return 111
    #     elif x == "transpose":
    #         return 112
    #     elif x == "hermitian":
    #         return 113
    #
    # TransA = trans_code(A_trans)
    # TransB = trans_code(B_trans)

    TransA = 111
    TransB = 111

    m = A.shape[1] if TransA == 112 or TransA == 113 else A.shape[0]
    n = B.shape[0] if TransB == 112 or TransB == 113 else B.shape[1]
    k = A.shape[0] if TransA == 112 or TransA == 113 else A.shape[1]
    Order = 101  # 101 for row-major, 102 for column major data structures
    alpha = 1.0
    beta = -1.0
    lda, ldb, ldc = k, n, n # problematic here, need to see doc
    C = np.zeros((m,n))

    mkl.cblas_dgemm(
        ctypes.c_int(Order),
        ctypes.c_int(TransA),
        ctypes.c_int(TransB),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_double(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(lda),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ldb),
        ctypes.c_double(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ldc))
    return C


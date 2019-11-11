import numpy as np
from numpy import ndarray
from scipy.special import erfinv

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


def mldivide(A: ndarray, B: ndarray):
    res: ndarray = np.linalg.solve(A.transpose().conj(), B.transpose().conj())
    return res.transpose().conj()


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


def receive_signal(p, tau_p, H, gaussian_rng=None):
    if gaussian_rng is None:
        gaussian_rng = np.random.randn
    elif gaussian_rng == 'debug':
        gaussian_rng = randn2

    Np = np.sqrt(0.5) * (
            randn2(realization_cnt, K, L, f) + 1j * randn2(M, realization_cnt, K, L, f)
    )




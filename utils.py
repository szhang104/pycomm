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

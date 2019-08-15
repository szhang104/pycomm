import numpy as np
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
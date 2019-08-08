import numpy as np
from scipy.linalg import toeplitz
from scipy.integrate import quad
from numba import jit, njit, prange

# TODO improve runspeed. very slow compared to MATLAB

@njit
def correlation_real(x, antenna_spacing, col):
    return np.cos(2 * np.pi * antenna_spacing * col * np.sin(x))

@njit
def correlation_imag(x, antenna_spacing, col):
    return np.sin(2 * np.pi * antenna_spacing * col * np.sin(x))

@njit
def gaussian_pdf(x, mean, dev):
    return np.exp(-(x-mean) ** 2 / (2 * dev ** 2)) / (np.sqrt(2 * np.pi) * dev)

@njit
def laplace_pdf(x, mean, scale):
    return np.exp(-np.abs(x-mean)/scale) / (2 * scale)

@njit
def uniform_pdf(x, a, b):
    return 1/(b-a)


@njit
def corr(x, theta, asd, antenna_spacing, dist, col, real_imag):
    if real_imag == 0:
        res = np.cos(2 * np.pi * antenna_spacing * col * np.sin(x))
    else:
        res = np.sin(2 * np.pi * antenna_spacing * col * np.sin(x))
    if dist =='gaussian':
        res *= gaussian_pdf(x, theta, asd)
    elif dist =='laplace':
        res *= laplace_pdf(x, theta, asd/np.sqrt(2))
    elif dist == 'uniform':
        res *= uniform_pdf(x, theta-np.sqrt(3)*asd, theta+np.sqrt(3)*asd)
    return res


def R_local_scattering(M, theta, asd_deg,
                       antenna_spacing=0.5,
                       dist='Gaussian',
                       accuracy=1,
                       dtype=np.complex128):
    """
    Generate the spatial correlation matrix for the local scattering model,
    defined in (2.23) for different angular distributions.

    :param M: # of antennas
    :param theta: nominal angle in radians
    :param asd_deg: # angular standard dev. around the nominal angle, in degrees
    :param antenna_spacing: opt., spacing b/w antennas in wavelengths
    :param dist: opt., angular distribution. 'Gaussian' by default. Valid
    values include 'Gaussian', 'Uniform', 'Laplace'
    :param accuracy: whether to use approximation
    :return: R: (M, M) spatial correlation matrix
    """
    asd = asd_deg * np.pi / 180 # in radians

    # correlation matrix is Toeplitz structure, so only need first row
    first_row = np.zeros([M,], dtype=dtype)

    lb = None
    ub = None

    dist = dist.lower()
    if dist == 'gaussian':
        # dist_obj = norm(loc=theta, scale=asd)
        lb = theta - 20 * asd
        ub = theta + 20 * asd
    elif dist == 'uniform':
        # [-sqrt(3)*asd_deg, +sqrt(3)*asd_deg]
        # dist_obj = uniform(loc=theta-np.sqrt(3)*asd,
        #                    scale=2*np.sqrt(3)*asd)
        lb = theta-np.sqrt(3)*asd
        ub = theta+np.sqrt(3)*asd
    elif dist == 'laplace':
        # dist_obj = laplace(loc=theta, scale=asd/np.sqrt(2))
        lb = theta - 20 * asd
        ub = theta + 20 * asd
    else:
        raise NotImplementedError



    for col in range(0, M):
        # distance from the first antenna
        c_real:float = quad(
            func=corr,
            a=lb,
            b=ub,
            args=(theta, asd, antenna_spacing, dist, col, 0)
        )[0]
        c_imag:float = quad(
            func=corr,
            a=lb,
            b=ub,
            args=(theta, asd, antenna_spacing, dist, col, 1))[0]

        first_row[col] = complex(c_real, c_imag)

    return toeplitz(c=first_row.conjugate())
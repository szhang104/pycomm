import numpy as np
from matplotlib import pyplot as plt
import time
from local_scatter import R_local_scattering


def plot_eigs_R():
    """
    plots the eigenvalues  from the highest to lowest, comparing the different
    angular distributions (Laplace, Gaussian, Uniform).
    See plot 2.6 in the book
    Returns
    -------
        None
    """
    starttime = time.time()
    M = 100
    theta = np.pi / 10
    asd = 10
    spacing = 0.5
    dists = ['Gaussian', 'Uniform', 'Laplace', 'Uncorrelated']
    styles = ['k', 'b-.', 'r--', 'k:']
    Rs = {}
    eigs = {}
    for dist, style in zip(dists, styles):
        if dist == 'Uncorrelated':
            Rs[dist] = np.eye(M)
        else:
            Rs[dist] = R_local_scattering(M, theta, asd, spacing, dist=dist)
        eig = np.linalg.eig(Rs[dist])[0].real
        eig = np.sort(eig)[::-1]
        eig[eig < 0] = 1.0e-10
        eigs[dist] = eig

        plt.plot(np.arange(1, 101), 10 * np.log10(eigs[dist]), style, label=dist)
        plt.legend()
        plt.ylim(bottom=-50, top=10)

    plt.show()
    print('finished in {} secs'.format(time.time() - starttime))

if __name__ == '__main__':
    plot_eigs_R()

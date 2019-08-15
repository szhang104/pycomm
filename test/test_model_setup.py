
import numpy as np
from scipy.io.matlab import loadmat
from model_setup import channel_stat_setup
from channel_estimation import channel_estimates

def test_model_setup():
    np.random.seed(12345)
    R, channel_gain = channel_stat_setup(K=10, L=16, M=100, asd_degs=[10,])
    data_m = loadmat('test_4_5.mat')
    R_m = data_m['R']
    channel_gain_m = data_m['channelGaindB']

    assert np.allclose(channel_gain_m, channel_gain)
    assert np.allclose(R.squeeze(), R_m)
    return R.squeeze()


def test_channel_estimation():
    data_m = loadmat('test_4_5.mat')
    R_m = data_m['R']
    channel_gain_m = data_m['channelGaindB']


    B = 20e6
    noiseFigure = 7
    noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

    M = 10
    K = 10
    L = 16
    rel_cnt = 100
    np.random.seed(12345)
    Hhat, C, tau_p, Rscaled = channel_estimates(R_m[0:M, 0:M, ],
                      channel_gain_db=channel_gain_m - noiseVariancedBm,
                      realization_cnt=rel_cnt,
                      M=M,
                      K=K,
                      L=L,
                      p=100,
                      f=1)
    assert np.allclose(Hhat, data_m['Hhat'])
    assert np.allclose(C, data_m['C'])
    assert np.allclose(Rscaled, data_m['Rscaled'])




import numpy as np
from scipy.io.matlab import loadmat
from model_setup import channel_stat_setup

def test_model_setup():
    np.random.seed(12345)
    R, channel_gain = channel_stat_setup(K=10, L=16, M=100, asd_degs=[10,])
    data_m = loadmat('test_4_5.mat')
    R_m = data_m['R']
    channel_gain_m = data_m['channelGaindB']

    assert np.allclose(channel_gain_m, channel_gain)
    assert np.allclose(R.squeeze(), R_m)
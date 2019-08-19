
import numpy as np
from scipy.io.matlab import loadmat
from model_setup import channel_stat_setup
from channel_estimation import channel_estimates
from SE_UL import SE_UL
from utils import noise_variance_db
from pandas import DataFrame
# import xarray

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

def test_SE_UL():
    np.random.seed(12345)
    data_m = loadmat('test_4_5.mat')
    res_m = loadmat('test_4_5_results.mat')
    R = data_m['R']
    channel_gain_m = data_m['channelGaindB']
    setup_cnt = 1 # number of UE location settings
    realization_cnt = 100
    B = 20e6
    K = 10
    L = 16
    p = 100 # in mW
    tau_c = 200 # length of coherence block
    methods = ['MR', 'RZF', 'MMMSE', 'ZF', 'SMMSE']
    sumSE = {}
    M_list = np.arange(10, 40, 10)
    f_list = [1, 2, 4]
    for method in methods:
        sumSE[method] = np.zeros((len(M_list), len(f_list), realization_cnt))

    for i in range(setup_cnt):
        print('Realization {i}')
        channel_gain_over_noise = channel_gain_m - noise_variance_db(B)
        for m_idx, m in enumerate(M_list):
            print('Antenna count: {}'.format(m))
            for f_idx, f in enumerate(f_list): # pilot reuse factor
                print('reuse {}'.format(f))
                Hhat, C, tau_p, Rscaled = channel_estimates(R[0:m, 0:m],
                                  channel_gain_over_noise,
                                  realization_cnt,
                                  m, K, L, p, f)

                # each value is shape (M, L)
                SE = SE_UL(Hhat, C, Rscaled, tau_c, tau_p, realization_cnt, m, K, L, p)
                # save the sum only
                for method in methods:
                    sumSE[method][m_idx, f_idx, i] = np.average(np.sum(SE[method], axis=0))
                    from_matlab = res_m['sumSE_'+method]
                    if len(from_matlab.shape) == 2:
                        comp = from_matlab[m_idx, f_idx]
                    else:
                        comp = from_matlab[m_idx, f_idx, i]
                    assert np.allclose(sumSE[method][m_idx, f_idx, i], comp)





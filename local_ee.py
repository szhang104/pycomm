import jax
from ee_max import setup, zf
import jax.numpy as np


def get_total_power(W, R=0.0):
    M = W.shape[-1]
    P_fixed = 1
    P_per_antenna = 0.1
    coeff1 = 1e-9
    return transmit_power(W).sum() + P_per_antenna * M + P_fixed + coeff1 * R

def transmit_power(W):
    return np.sum(np.abs(W) ** 2.0, axis=1)

def user_rate(H, W, B):
    HHW2 = np.abs(H.conj() @ W.T) ** 2.0
    p_sig = np.diag(HHW2)
    p_int = HHW2 - np.diag(p_sig)
    SINR = p_sig / (p_int.sum(axis=1) + 1.0)
    rate = B * np.log2(1 + SINR)
    return rate

def EE_d(W, H, B):
    R = user_rate(H, W, B).sum()
    return R / get_total_power(W, R)


if __name__ == "__main__":
    K, M = 8, 80
    P_noise = 1e-3 * 10 ** (0.1 * -94)
    B = 20 * 1e6 # 20MHz bandwidth
    P_user_max=0.1
    H = setup(K, M, seed=12345)[0] / np.sqrt(P_noise)
    W_zf = jax.numpy.sqrt(P_user_max) * zf(H)
    EE_grad = jax.jacrev(EE_d, 0)
    res = EE_grad(W_zf, H, B)
    print(res)
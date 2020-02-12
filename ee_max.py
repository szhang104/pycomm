import numpy as np
import scipy as sp
import cvxpy as cp
import dccp

def t_posdef():
    # testing if the matrix B = A^T A, A = [  h_R  h_I ]
    #                                      [ -h_I  h_R ]
    # is positive definite
    K, M = 5, 3
    H = np.random.randn(K, M) + 1j * np.random.randn(K, M)  # K, M

    gamma = 1e-5
    A = []
    B = []
    for i in range(K):
        A.append(np.zeros((2, 2 * K * M)))
        A[i][0, i * 2 * M: 2 * M * (i + 1)] = np.concatenate(
            [H[i].real, H[i].imag])
        A[i][1, i * 2 * M: 2 * M * (i + 1)] = np.concatenate(
            [-H[i].imag, H[i].real])
        B.append(A[i].T @ A[i])
        eigs = np.linalg.eigvalsh(B[i])
        print(eigs)
    BB = B[0] - gamma * sum(B[1:])
    print(np.linalg.cholesky(BB))


def t_qcqp():
    # Generate a random non-trivial quadratic program.
    m = 15
    n = 10
    p = 5
    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
                      [G @ x <= h,
                       A @ x == b])
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)


def t_dccp():
    x = cp.Variable(2)
    y = cp.Variable(2)
    myprob = cp.Problem(cp.Maximize(
        cp.norm(x - y, 2)),
        [0 <= x, x <= 1, 0 <= y, y <= 1])
    print("problem is DCP:", myprob.is_dcp())  # false
    print("problem is DCCP:", dccp.is_dccp(myprob))  # true
    result = myprob.solve(method='dccp')
    print("x =", x.value)
    print("y =", y.value)
    print("cost value =", result[0])


def t_cvxpy_cgrad():
    # seems not working due to type conversion from complex128 to float64
    # in set_dense_data
    # return _cvxcore.LinOp_set_dense_data(self, matrix)
    # TypeError: Cannot cast array data from dtype('complex128') to dtype(
    #   'float64') according to the rule 'safe'
    n = 2
    x = cp.Variable(n, complex=True)
    h = np.random.randn(2) + 1j * np.random.randn(2)
    expr = h.conj() @ x
    x.value = np.array([1.0 + 1.0j, 1.0 + 1.0j])
    y = cp.Variable((5, 2))
    ee = cp.sum(cp.power(y, 2))
    y.value = np.arange(10).reshape(5, 2)
    expr.grad


def to_real(x):
    """
    Return a real-valued array whose last dimension is the real and imaginary
    parts of the input
    :param x: complex valued array
    :return:
    """
    return np.stack([x.real, x.imag], axis=-1)


def to_imag(x):
    """
    Return a complex-valued array whose last dimension is the real and
    imaginary parts of the input
    :param x: real valued array (..., 2)
    :return:
    """
    return x[:, :, 0] + 1j * x[:, :, 1]


def cnorm2(w):
    # treating w as a (n, 2) matrix of real values
    return cp.sum(cp.power(w, 2))


def cahb_r(a, b):
    # a, b are real with shape (n, 2)
    # returns the real part of the inner product a^H b
    return a[:, 0] @ b[:, 0] + a[:, 1] @ b[:, 1]


def cahb_i(a, b):
    # a, b are real with shape (n, 2)
    # returns the real part of the inner product a^H b
    return a[:, 0] @ b[:, 1] - a[:, 1] @ b[:, 0]


def cahb2(a, b):
    return cahb_r(a, b) ** 2.0 + cahb_i(a, b) ** 2.0


def min_sinr_req(min_sinr, h, w, w_all, P_other=0.0):
    # h, w are real
    # h is normalized wrt noise: h = h / sigma
    # h is the channel state of a single user: (M, 2) M is antenna count
    # w is the precoding vec (M, 2)
    # w_other is an iterable of all w that can have an interference
    sig = cahb_r(h, w)
    intf_noise = [cahb_r(h, w_j) for w_j in w_all] + [cahb_i(h, w_j) for
                                                      w_j in w_all]
    intf_noise.append(1.0)  # normalized noise
    intf_noise.append(np.sqrt(P_other))
    t = np.sqrt(1 + 1.0 / min_sinr) * sig
    # rhs = np.sqrt(all_power + 1.0 + other)
    cons_soc = cp.SOC(
        t,  # the scalar
        cp.vstack(intf_noise)  # a vector, whose 2-norm is leq t
    )
    return cons_soc


def min_power(H, P_user_max, SINR_user_min):
    """
    Solve the minimum single cell power given the minimum SINR required
    :param H: input channel state. (K, M) complex
    :param P_user_max: maximum transmit power per user. the norm of w
    :param SINR_user_min: minimum per user SINR, in absolute number
    :return: W
    """
    def w_numpy(W):
        res = np.stack([x.value for x in W])
        return to_imag(res)

    H_r = to_real(H)
    K, M = H.shape[0], H.shape[1]
    users, antennas = range(K), range(M)
    W = []
    for _ in users:
        W.append(cp.Variable((M, 2)))  # the equivalent real variables

    cons_power = []
    cons_sinr = []
    cons_force_real = []
    # power
    for u in users:
        cons_power.append(
            cnorm2(W[u]) <= P_user_max[u])

    # min SINR req
    # used a common trick to add these constraints as second-order cone
    # constraint which is convex
    for u in users:
        cons_sinr.append(
            min_sinr_req(SINR_user_min[u], H_r[u], W[u], W))

    # for the above to work, all h_i^H w_i must be real-valued
    # add these linear equality constraints so no big deal
    for u in users:
        cons_force_real.append(
            cahb_i(H_r[u], W[u]) == 0.0)

    total_power = cp.sum([cnorm2(ww) for ww in W])

    prob = cp.Problem(
        cp.Minimize(total_power),
        cons_power + cons_sinr + cons_force_real
    )
    prob.solve()
    sol = None
    if prob.status in ["optimal", "optimal_inaccurate"]:
        sol = w_numpy(W)
    return prob.status, sol


def maxee(H,
          P_user_max=1e-1,
          SINR_user_min=10**(8/10), # 8dB
          max_iter=200,
          eps=1e-3,
          eta=2*1e-2,
          P_fixed=5.0,
          B=20*1e6):



    def delta_rate_to_delta_SINR(delta_rate, cur_SINR):
        return (2.0 ** (delta_rate / B) - 1) * (1 + cur_SINR)

    def report():
        print("============Iter {}===============".format(it))
        print("EE: {} Mbit/ J".format(EE / 1e6))
        print("User rates in Mbits/s: {}\n{}".format(
            total_rate / 1e6, user_rate_v / 1e6))
        print("User powers in mW: {}\n{}".format(total_power * 1e3,
                                                 user_power_v * 1e3))
        if it > 0:
            print("rate change: {}, power change: {}, ratio: {}".format(
                (total_rate - o_total_rate) / 1e6,
                (total_power - o_total_power) * 1e3,
            (total_rate - o_total_rate) / (total_power - o_total_power) / 1e6
            ))


    K, M = H.shape[0], H.shape[1]
    P_req = P_user_max * np.ones(K)
    SINR_req = SINR_user_min * np.ones(K)
    for it in range(max_iter):
        status, W = min_power(H, P_req, SINR_req)
        if status not in ["optimal", "optimal_inaccurate"]:
            print("The sub-problem is {}".format(status))
            break
        if it > 0:
            o_total_rate, o_total_power, old_EE, old_W = total_rate, \
                                                      total_power, EE, W
        user_rate_v = user_rate(H, W, B)
        user_power_v = user_power(W)
        total_rate, total_power = user_rate_v.sum(), user_power_v.sum() + P_fixed
        EE = total_rate / total_power
        if it > 0 and EE < old_EE:
            break
        report()
        j = np.argmax(abs(user_power_v - P_user_max)) # the user with the
        # most slack power budget
        # now add more SINR
        to_add_r = eta * total_rate
        # P_req = user_power_v
        delta_SINR_j = delta_rate_to_delta_SINR(to_add_r, SINR_req[j])
        SINR_req[j] += delta_SINR_j
        # SINR_req -= delta_SINR_j / K
    return old_W


def db2real_power(x):
    return 10 ** ( x / 10 )


def user_rate(H, W, B):
    HHW2 = np.abs(H.conj() @ W.T) ** 2.0
    p_sig = np.diag(HHW2)
    p_int = HHW2 - np.diag(p_sig)
    SINR = p_sig / (p_int.sum(axis=1) + 1.0)
    rate = B * np.log2(1 + SINR)
    return rate


def user_power(W):
    return np.sum(np.abs(W) ** 2.0, axis=1)


def large_scale_fading(d,
                       gain_1km_db=-148.1,
                       alpha=3.76,
                       has_shadow_fading=True,
                       shadow_fading_var=100.0):
    # d is a 1-d array or a scalr, the distances of the receivers
    if np.isscalar(d):
        n = None
    else:
        n = d.shape[-1]
    gain_db = gain_1km_db \
              - 10 * alpha * np.log10(d / 1000)
    if has_shadow_fading:
        shadow_fading = np.random.normal(0.0, shadow_fading_var, n)
        gain_db += shadow_fading # shadow fading
    return db2real_power(gain_db)


def uncorrelated_rayleigh(K, M, beta, inst=1):
    # N(0, I)
    H = 1 / np.sqrt(2) * (
            np.random.randn(inst, K, M) +
            1j * np.random.randn(inst, K, M))
    H = np.diag(np.sqrt(beta)) @ H
    return H


def test_uncorrelated_rayleigh():
    K, M = 3, 4
    beta= np.array([1.0, 2.0, 3.0])
    H = uncorrelated_rayleigh(K, M, beta, inst=2000)
    H = np.expand_dims(H, axis=3)
    R = np.mean(H.conj() @ H.transpose((0,1,3,2)), axis=0)
    return R


def zf(H):
    A = H.conj().T @ H
    B = H
    # solve x A = B
    return np.linalg.solve(A, B.conj().transpose()).conj().transpose()


def complex_normalize(X, axis=-1):
    """
    Normalize the complex n-dim array on the dimension axis
    Parameters
    ----------
    X: n-dimension complex array
    Returns
    -------
    """
    mags = np.linalg.norm(np.abs(X), axis=axis, keepdims=True)
    return X / mags

if __name__ == "__main__":
    K, M = 10, 40
    P_user_max = 0.1  #
    SINR_user_min = 10 ** (7 / 10)   # 7dB
    B = 20*1e6
    P_fixed = 5.0
    # should be -101dBm for 20MHz bandwidth
    # P_noise = 1e-3 * 10 ** (0.1 * (-174 + 10 * np.log10(B)))
    # or as in bjornsson book, set to -94 dBm
    P_noise = 1e-3 * 10 ** (0.1 * -94)


    d = np.random.uniform(35, 124, size=(K,)) # cell radius 125m, minimum
    print(d)
    # dist with BS is 35m
    beta = large_scale_fading(d, has_shadow_fading=False)

    H = uncorrelated_rayleigh(K, M, beta) # first dimension is instance, K, M
    snr = 10 * np.log10(np.power(np.abs(np.sqrt(P_user_max) * H), 2).mean(
        axis=-1) / P_noise)

    for H_ in H:
        H_ = H_ / np.sqrt(P_noise)
        # Must normalize H against noise power before use!!
        W_ee = maxee(H_,
                  P_user_max=P_user_max,
                  SINR_user_min=SINR_user_min,
                  B=B)
        W_zf = np.sqrt(P_user_max) * complex_normalize(zf(H_))
        P_zf = user_power(W_zf) + P_fixed
        R_zf = (user_rate(H_, W_zf, B))
        print(R_zf.sum()/P_zf.sum()/1e6)

    # H_norm = np.sqrt((np.abs(H) ** 2).sum(axis=1))
    #
    # W_MR = P_user_max * (H.T / H_norm).T
    # HHW = np.abs(H @ W.conj().T) ** 2 / P_noise
    # HHW_MR = np.abs(H @ W_MR.conj().T) ** 2 / P_noise

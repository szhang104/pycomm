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


def maxee(H, P_user_max=1e-3, SINR_user_min=1e2):
    """
    Solve max ee
    :param H: input channel state. (K, M) complex
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
            cnorm2(W[u]) <= P_user_max)

    # min SINR req
    # used a common trick to add these constraints as second-order cone
    # constraint which is convex
    for u in users:
        cons_sinr.append(
            min_sinr_req(SINR_user_min, H_r[u], W[u], W))

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

    if prob.status == ["infeasible", "unbounded"]:
        raise RuntimeError("The sub-problem is {}".format(prob.status))
    else:
        print(prob.status, prob.value)
        sol = w_numpy(W)
        for x in cons_power:
            print(x.dual_value)
        return sol


if __name__ == "__main__":
    K, M = 20, 100
    P_user_max = 1e-3 * 10 ** (-1)  # -10 dBm
    SINR_user_min = 1e2  # 20dB
    P_noise = 1e-3 * 10 ** (-10.4)  # -104dbm

    H = 1 / np.sqrt(2) * (np.random.randn(K, M) + 1j * np.random.randn(K, M))
    H = 1e-4 * H  # channel gain is -80 dB
    snr = 10 * np.log10(np.power(np.abs(np.sqrt(P_user_max) * H), 2).mean(
        axis=1) /
                        P_noise)

    W = maxee(H / np.sqrt(P_noise), P_user_max=P_user_max,
              SINR_user_min=SINR_user_min)
    H_norm = np.sqrt((np.abs(H) ** 2).sum(axis=1))

    W_MR = P_user_max * (H.T / H_norm).T
    HHW = np.abs(H @ W.conj().T) ** 2 / P_noise
    HHW_MR = np.abs(H @ W_MR.conj().T) ** 2 / P_noise

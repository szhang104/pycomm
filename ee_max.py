import numpy as np
import scipy as sp
import cvxpy as cp


def t_posdef():
    # testing if the matrix B = A^T A, A = [  h_R  h_I ]
    #                                      [ -h_I  h_R ]
    # is positive definite
    K, M = 5, 3
    H = np.random.randn(K, M) + 1j * np.random.randn(K, M) #K, M

    gamma = 1e-5
    A = []
    B = []
    for i in range(K):
        A.append(np.zeros((2, 2 * K * M)))
        A[i][0, i*2*M: 2*M*(i+1)] = np.concatenate([H[i].real, H[i].imag])
        A[i][1, i*2*M: 2*M*(i+1)] = np.concatenate([-H[i].imag, H[i].real])
        B.append(A[i].T @ A[i])
        eigs = np.linalg.eigvalsh(B[i])
        print(eigs)
    BB = B[0] - gamma * sum(B[1:])
    print(np.linalg.cholesky(BB))



def maxee(H, P_user_max=1e-4, R_user_min=10.0):
    """
    Solve max ee
    :param H: input channel state. (K, M) complex
    :return: W
    """
    K, M = H.shape[0], H.shape[1]
    users, antennas = range(K), range(M)
    W = cp.Variable((K, M), imag=True)

    H_W = H.conj() @ W.T # (K, K)
    H_W = H_W * H_W.conj() # (K, K)


    def cnorm(w):
        return cp.norm(w, 2)

    cons = []
    for u in users:
        cons.append(
            cnorm(W[u]) <= (P_user_max))

    for u in users:
        p_sig = H_W[u,u]
        p_int = H_W[u].sum() - p_sig
        cons.append(
            cp.log(1 + p_sig / p_int) / cp.log(2) >= R_user_min)

    return




def t_qcqp():
    # Generate a random non-trivial quadratic program.
    m = 15
    n = 10
    p = 5
    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T@P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G@np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T@x),
                      [G@x <= h,
                       A@x == b])
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)



if __name__ == "__main__":
    t_posdef()
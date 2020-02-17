import jax
import jax.numpy as np
import numpy as onp
from qpsolvers import solve_qp
from scipy.optimize import minimize

def abs2(x):
    return x.real ** 2.0 + x.imag ** 2.0


def aH_dot_x_square(x, a):
    # x, a are complex vectors
    return abs2(a.conj() @ x)


def aH_dot_x_square_real(x_r, x_i, a_r, a_i):
    return (a_r @ x_r + a_i @ x_i) ** 2.0 + (a_r @ x_i - a_i @ x_r) ** 2.0


def hessian(f, argnums = 0):
    return jax.jacfwd(jax.jacrev(f, argnums), argnums)





def test_gd_complex(alpha=0.1, max_iter=100):
    # sanity test to see if jax autograd gives the right descent direction
    # Never forget to conj() the returned grad if the complex function is real-valued
    # see https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#complex-numbers
    ww = jax.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    ww *= 10.0
    tt = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    gk = lambda x: abs2(x-tt).sum()
    gkg = jax.value_and_grad(gk)

    for i in range(max_iter):
        val, grad = gkg(ww)
        grad = grad.conj()
        print("iter {}".format(i))
        print(val)
        print(ww-tt)
        ww -= alpha * grad


def test_gd_w(K=4, M=12, gamma_db=10, alpha=0.1, max_iter=5):
    # the problem P1 in bjornson paper, min power precoding subject to
    # individual user rate
    # minimize_{w} \sum_k ||w_k||^2
    # s.t. SINR_k \geq \gamma_k

    def total_power(W):
        # W is complex (K, M)
        return abs2(W).sum()

    def SINR(W, H):
        # H: (K, M)
        # W: (K, M)
        all_powers = abs2(H.conj() @ W.transpose()) # (K, K), all powers
        signal_powers = np.diag(all_powers)
        intra_intf = all_powers.sum(axis=1) - signal_powers
        sinr = signal_powers / (intra_intf + 1.0)
        return sinr

    def optimum(H, l=1, P=1e3):
        # regularized ZF
        # H: (K, M)
        # l is the average user transmit power normalized to P_noise
        # P is the total transmission power normalized to P_noise
        W = np.identity(K) + l * H @ H.conj().T
        p = np.linalg.solve(W, H)
        P_sqrt = np.diag(p)
        return H.T @ np.linalg.solve(W, P_sqrt)


    def real2comp(x):
        return x[:len(x)//2] + 1j * x[len(x)//2:]

    def comp2real(z):
        return np.concatenate((np.real(z), np.imag(z)))


    def gd(f, f_grad, x_0, step=0.01, args=None, max_iter=500, debug=False):
        x = x_0
        step0 = step
        delta_x = lambda x: -step * (np.conj(fg))
        while True:
            step = step0
            fg = f_grad(x, *args)
            val = f(x, *args)

            # line search for a good step size

            while f( x + delta_x(x), *args) >= val:
                step = step / 2
            x += delta_x(x)

            delta_x_mean = np.abs(delta_x(x)).mean()
            if debug:
                print(val, delta_x_mean)

            if delta_x_mean < 1e-6:
                break

        return x

    def solve(H, gamma, step=0.05, method="np"):
        # need to find a feasible initial!!
        W = onp.random.randn(K, M) + 1j * onp.random.randn(K, M)
        u = 10 * np.ones(K,)

        def lagrangian(W, u, H):
            return total_power(W) \
                   + u @ (gamma - SINR(W, H)) \
                   + 50000 * abs2(gamma-SINR(W, H)).sum()

        def lagrangian_real_flat(x, u, H):
            c_x = real2comp(x)
            return lagrangian(c_x.reshape(K, M), u, H)

        def lagrangian_grad_real_flat(x, u, H):
            z = real2comp(x).reshape(K, M)
            gg = jax.jacrev(lagrangian)
            grad = gg(z, u, H).conj()
            return comp2real(grad.reshape(K*M,))


        SINR_g = jax.jacrev(SINR, 0)
        power_vng = jax.value_and_grad(total_power)
        sinr = SINR(W, H)
        sinr_grad = SINR_g(W, H)
        lag_grad = jax.jacrev(lagrangian)
        # W_opt = optimum(H)

        for i in range(100):
            print("iter {}".format(i))
            power, power_grad = power_vng(W)
            print("obj: {}, SINR: {}".format(power, sinr))
            print("lag: {}".format(lagrangian(W, u, H)))
            W_0 = onp.random.randn(K, M) + 1j * onp.random.randn(K, M)
            if method == "gd":
                res = gd(lagrangian, lag_grad, W_0, args=(u, H), step=0.001, debug=True)
                new_x = res
            elif method == "np":
                res = minimize(total_power,
                               comp2real(W_0.reshape(K*M,).copy()),
                               args=(u, H),
                               jac=lagrangian_grad_real_flat,
                               )
                new_x = res.x
            W = real2comp(new_x).reshape(K, M)
            # res = minimize(lagrangian,
            #                W_0.copy(),
            #                args=(u, H),
            #                jac=lag_grad
            #                )
            sinr = SINR(W, H)
            u -= step * (gamma - sinr)
            print(u)

    H = 1/ np.sqrt(2) * (onp.random.randn(K, M) + 1j * onp.random.randn(K, M))
    gamma = 10 ** (gamma_db / 10)
    solve(H, gamma, method="np")





CONFIG = {
    "p_n": 1e-9, # receiver noise power
    "p_bs_max": 1e-3, # max bs power, including signal processing and
    # transmission
    "p_fixed": 1e-5, # fixed bs power
    "p_ant": 1e-5, # per BS antenna power
    "p_sp": 1e-5, # per unit throughput power needed for signal processing
    "r_min": 4e8, # minimum user rate
}


# Indicator functions to use as constraints
# return value non-positive as satisfying the constraint
# gradient information derived from these functions
def user_int_plus_noise(w_intra, phi, t, h):
    # with respect to a certain user bk
    # w_intra is (K-1, N_ant) complex vector containing all the precoding vec
    # of K-1 co-cell users. The caller needs to track the exact id's of these
    # phi is the total received inter-cell interference
    p_intra = np.sum(
        aH_dot_x_square(w, h) for w in w_intra)
    return phi + p_intra + CONFIG["p_n"] - t


def cell_power(w_b, v_b, c_b):
    # w_b is the precoding used by cell b
    # C^{K, M}
    p_trans = abs2(w_b).sum()
    return p_trans \
           + CONFIG["N_b"] * CONFIG["P_ant"] \
           + v_b * CONFIG["P_sg"] \
           + CONFIG["P_fixed"] - c_b


def user_sinr(s, t, h, w):
    p_sig = aH_dot_x_square(w, h)
    return s * t - p_sig


def cell_rate_ub(u, s_b):
    # s_b is a vector of shape (K,), containing all LB of SINR of this cell's
    # users
    return u - sum(janp.log(1 + s) for s in s_b)


def cell_rate_lb(v, s_b):
    return sum(janp.log(1+s) for s in s_b) - v


def cell_ee_lb(r, c, u):
    return r * c - u


def cell_power_ub(c):
    return c - CONFIG["p_bs_max"]


def cell_give_take_intf(Ψ_all, Φ, D):
    # Ψ_all: (N_c, N_c K)
    # Φ: (K, )
    # D: (K, N_c K)
    return (D @ Ψ_all.transpose()).sum(axis=1) - Φ


class BS_State:
    def __init__(self, i, H):
        self.id = i
        self.w = H[id, id] # initial precoding set to maximum combining
        # initial interference from other cells
        self.Φ = 1e-7 * np.ones(H.shape[2])
        self.u = 0.0
        self.v = 10.0
        self.c = 10.0
        self.t = 1e-6 * np.ones(H.shape[2])
        self.r = 0.02


def solve_sub():
    return

def negotiate(res):
    return


def solve(H, max_iter=100):
    # H: (N_c, N_c, K, N_b)
    # add N_b selection
    N_b = H.shape[0]
    BS_States = [BS_State(i, H) for i in range(N_b)]

    for it in range(max_iter):
        # phase 1: each BS solve its own sub problem without the global
        # constraint
        # TODO: parallelize this
        for i in range(N_b):
            res[i] = solve_sub(BS_States[i]) # a candidate solution, simply
            # the local optimum
        res = negotiate(res)
        # print some results




class Constraint:
    def __init__(self, f, argnums=0):
        self.f = f
        self.argnums = argnums
        self.vg = jax.value_and_grad(self.f, argnums)
        self.hess = hessian(self.f)

    def run(self, *args, **kwargs):
        return self.vg(*args, **kwargs)

    def run_hess(self, *args, **kwargs):
        return self.hess(*args, **kwargs)




ex727_e2 = Constraint(
    lambda x: 0.05882 * x[0] * x[2] + 0.1 * x[3] - 1
)

ex727_e3 = Constraint(
    lambda x: 4 * x[1] / x[2] + 2/(x[1] ** 0.71 * x[2])
              + 0.05882 * x[0] / x[1] ** 1.3 - 1
)

ex727_obj = lambda x: 0.4 * x[3] ** 0.67 / x[0] ** 0.67 - x[3]
# optimum:
x_opt = [0.5652954363,0.6164017066,5.6370006246,8.1256977401]

def solver(obj, constraints):
    max_iter = 1
    x = np.array([0.1, 0.1, 0.1, 0.1])
    obj_hess_f = hessian(ex727_obj)
    dual = np.array([5.0, 5.0])

    for it in range(max_iter):
        cons_grad = []
        cons_val = []
        cons_hess = []
        for cons in constraints:
            val, grad = cons.run(x)
            cons_val.append(val)
            cons_grad.append(grad)
            cons_hess.append(cons.run_hess(x))
        jacobian = np.array(cons_grad)
        cons_val = np.array(cons_val)
        obj_hess = obj_hess_f(x)
        cons_hess = np.array(cons_hess)
        solve_qp(obj_hess + cons_hess.sum(axis=0), )



if __name__ == "__main__":
    test_gd_w()





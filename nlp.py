import jax
import jax.numpy as janp
import numpy as np
from qpsolvers import solve_qp


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
    ww = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
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
    p_intra = np.sum(
        aH_dot_x_square(w, h) for w in w_intra)
    return phi + p_intra + CONFIG["p_n"] - t

def cell_power(w_b, v_b, c_b):
    # w_b is the precoding used by cell b
    # C^{K, M}
    p_trans = abs2(w_b).sum()
    return p_trans + CONFIG["N_b"] * CONFIG["P_ant"] + v_b * CONFIG["P_sg"] -\
           c_b + CONFIG["P_fixed"]


def user_sinr(s, t, h_r, h_i, w_r, w_i):
    p_sig = aH_dot_x_square(w_r, w_i, h_r, h_i)
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
        id = i
        w = H[id, id] # initial precoding set to maximum combining
        Φ = 1e-7 * np.ones(H.shape[2]) # initial interference from other cells
        u = 0.0
        v = 10.0
        c = 10.0
        t = 1e-6 * np.ones(H.shape[2])
        r = 0.02



    def __add__(self, other):
        BS_State
        return


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





a = np.array([1+2j, 2-3j])
x = np.array([2+3j, 4-1j])

solver(ex727_obj, [ex727_e2, ex727_e3])





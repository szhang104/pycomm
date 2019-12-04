import numpy as np
import pyomo
import pyomo.environ as pe

def example_linear_frac(tol=1e-3, debug=True):
    model = pe.AbstractModel()
    model.q = pe.Param(default=0.5, mutable=True)
    model.goods = pe.Set(initialize=[0, 1])
    model.u_cost = pe.Param(model.goods, default=[1.0, 2.0])
    model.u_profit = pe.Param(model.goods, default=[4.0, 2.0])
    model.u_material = pe.Param(model.goods, default=[1.0, 3.0])
    model.material_avail = pe.Param(default=30.0)
    model.x = pe.Var(model.goods, domain=pe.NonNegativeReals)

    def cost(model):
        return pe.sum_product(model.u_cost, model.x) + 5

    def profit(model):
        return pe.sum_product(model.u_profit, model.x) + 10

    def cons_prod(model):
        return model.x[0] + 5 >= 2 * model.x[1]

    def cons_material(model):
        return pe.sum_product(model.u_material, model.x) \
               <= model.material_avail

    def getobj(model):
        return profit(model) - model.q * cost(model)

    # `rule` takes as the argument a function which returns a pyomo expression.
    # Doing this delays the expression building until the `AbstractModel` is
    # instantiated because operations like `sum_product` requires the arguments
    # to have concrete values.

    model.cons_prod = pe.Constraint(rule=cons_prod)
    model.cons_material = pe.Constraint(rule=cons_material)
    model.obj = pe.Objective(rule=getobj, sense=pe.maximize)

    def subsolve(q):
        model.q = q
        inst = model.create_instance()
        pe.SolverFactory("glpk").solve(inst)
        return inst

    q = 0.01
    while True:
        inst = subsolve(q)
        q = pe.value(profit(inst) / cost(inst))
        val = pe.value(inst.obj)
        if debug:
            print(q, val)
            print([pe.value(inst.x[g]) for g in inst.goods])
        if val < tol:
            if debug:
                print("Stop iteration because q_new < tol: {tol}")
            break
        else:
            pass



def max_sum_rate_opt(H):
    """
    Single-cell sum rate maximization, as an example to show that the
    analytical expression
    w_mmse_k =              + ∑_{i≠k} q_i h_i h_i^H)^{-1} h_k
                \sqrt{p_k} ---------------------------------------
                          ||(I + ∑_{i≠k} q_i h_i h_i^H)^{-1} h_k||
    H:
    complex CSI, shape is (K, M)


    Returns
    -------

    """
    model = pe.ConcreteModel()
    K, M = H.shape[0], H.shape[1]
    users = pe.RangeSet(K)
    antennas = pe.RangeSet(M)
    complex = pe.RangeSet(2)
    model.w = pe.Var(users, antennas, complex, domain=pe.Reals) # (K, M,
    # 2) real-value

    def comp_norm(x, y):
        return x * x + y * y

    def cons_norm(model, u):
        norm_u = pe.quicksum(comp_norm(model.w[u][a][0], model.w[u][a][1]) for
                           a in
                 antennas)
        return norm_u <= 1


    model.cons_norm = pe.Constraint(users, rule=cons_norm)








if __name__ == "__main__":
    example_linear_frac()








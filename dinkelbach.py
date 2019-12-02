import numpy as np
import pyomo
from pyomo.environ import *



def dinkelbach(P, Q, sub_solve, x_init=None, tol=1e-4, debug=False):
    """
    Maximize P(x) / Q(x)
    s.t. x in S

    Assumptions:
    -- Q(x) > 0 in the feasible set of x


    Parameters
    ----------
    P
    Q
    tol

    Returns
    -------

    """
    status = "Uninitialized"
    def get_sub_obj(q):
        return lambda x: P(x) - q * Q(x)

    def obj(x):
        return P(x) / Q(x)

    if not x_init:
        raise RuntimeError("x_init not supplied.")

    x = x_init
    q = obj(x)


    while True:
        # form the new instance of subproblem
        subobj = get_sub_obj(q)
        status, obj_val, x = sub_solve(q)
        if status == "Normal":
            q_new = obj(x_opt)
            if debug:
                print("new ratio: {q_new}")
            if q_new < tol:
                # DONE, this is the optimum
                if debug:
                    print("Stop iteration because q_new < tol: {tol}")
                status = "Optimum"
                break
            else:
                # otherwise keep the iteration
                q = q_new
                pass
        else:
            status = "Error"
            print("Stopped because of sub solver error.")

    return status, x




def example_linear_frac(tol=1e-3, debug=True):
    model = AbstractModel()
    model.q = Param(initialize=0.5)
    goods = Set(initialize=[0, 1])
    model.u_cost = Set(goods, initialize=[1, 2])
    model.u_profit = Param(goods, initialize=[4, 2])
    model.u_material = Set(goods, initialize=[1, 3])
    model.material_avail = Param(initialize=30.0)
    model.x = Var(goods, domain=NonNegativeReals)

    def cost(model):
        return sum_product(model.u_cost, model.x) + 5
    def profit(model):
        profit = sum_product(model.x, model.u_profit) + 10

    def cons_prod(model):
        return model.x[0] + 5 >= 2 * model.x[1]

    def cons_material(model):
        return sum_product(model.u_material,
                           model.x) <=model.material_avail
    def getobj(model):
        return profit(model) - model.q * cost(model)

    model.cons_prod = Constraint(rule=cons_prod)
    model.cons_material = Constraint(rule=cons_material)
    model.OBJ = Objective(rule=getobj)

    def subsolve(q):
        model.q = q
        inst = model.create_instance()
        SolverFactory("glpk").solve(inst)
        return inst

    q = 0.5
    while True:
        inst = subsolve(q)
        q = inst.ratio
        if q < tol:
            if debug:
                print("Stop iteration because q_new < tol: {tol}")
            break
        else:
            pass



if __name__ == "__main__":
    example_linear_frac()








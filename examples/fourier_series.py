from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy

from api import *


class Wave(eqx.Module):
    freq: float
    amplitude: float


class EmbedFourierSeries(DesignEmbedding):
    def __call__(self, x):
        return lambda t: sum(jnp.sin(2 * jnp.pi * freq * t) for freq in x)


class SimFourierSeries(DesignSimulation):
    def __call__(self, embedding, t):
        return embedding(t)


class EvalFourierSeries(DesignEvaluation):
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def __call__(self, state, points):
        loss = 0
        for p in points:
            loss += self.objective_function(state, p)
        return loss


class LBFGSDesignSearch(DesignSearch):
    def search(self, x, search_aux_data):
        grad = partial(self.gradient_function, eval_aux_data=search_aux_data)
        vals, treedef = jax.tree.flatten(x)
        res = scipy.optimize.minimize(
            self.value_function, vals, jac=self.gradient_function
        )
        return res.x


def lstsq(state, point):
    return (state[point.x] - point.y) ** 2


class Point(eqx.Module):
    x: int = eqx.field(static=True)
    y: float


p1 = Point(1, -20.0)
p2 = Point(2, 10.0)
p3 = Point(3, -0.5)
p4 = Point(4, -10.0)
points = [p1, p2]

design = []
for i in range(3):
    design.append(Wave(np.random.rand(), np.random.rand()))
sim_aux_data = np.linspace(0, 5, 6)
lr = 1e-3
epochs = 5000

embedding_module = EmbedFourierSeries()
sim_module = SimFourierSeries()
eval_module = EvalFourierSeries(lstsq)
jitted_grad_function = jax.jit(
    gradfunction(embedding_module, sim_module, eval_module, sim_aux_data)
)
design_search = LBFGSDesignSearch(eval_module, jitted_grad_function)

design_search.search(design, points)


# for epoch in range(epochs):
#     embedding = embedding_module(design)
#     state = sim_module(embedding, sim_aux_data)
#     val = eval_module.val(state=state)
#     grads = eval_module.grad(x=design, t=sim_aux_data)
#     design = design_search.search(design, grads, lr)
#
#     # print objective and gradient sum
#     if epoch % 100 == 0:
#         print(val, sum(grads))
#     # break if nan or gradient is trivial
#     if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < 0.01:
#         print(epoch, jnp.sum(grads), design)
#         break


# xnew = jnp.linspace(sim_aux_data[0], sim_aux_data[-1], 1000)
# plt.plot([o.x for o in objectives], [o.y for o in objectives], "bo")
# plt.plot(xnew, sim_module(embedding_module(design), xnew))
# plt.show()

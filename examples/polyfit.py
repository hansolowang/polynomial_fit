import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from api import *


class EmbedQuadratic(DesignEmbedding):
    def __call__(self, x):
        return lambda y: jnp.polyval(x, y)


class SimQuadratic(DesignSimulation):
    def __call__(self, embedding, t):
        return embedding(t)


class EvalQuadratic(DesignEvaluation):
    # TODO: support multiple objective functions?
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def __call__(self, state, points):
        loss = 0
        for p in points:
            loss += self.objective_function(state, p)
        return loss


class GradientDescent(DesignSearch):
    def search(self, x, search_aux_data):
        grads = self.gradient_function(x, search_aux_data)
        return x - lr * grads


def lstsq(state, point):
    return (state[point.x] - point.y) ** 2


class Point(eqx.Module):
    x: int = eqx.field(static=True)
    y: float


p1 = Point(1, -20.0)
p2 = Point(2, 10.0)
points = [p1, p2]

design = jnp.zeros(3, dtype="float32")
sim_aux_data = np.linspace(0, 5, 6)
lr = 1e-3
epochs = 2000

embedding_module = EmbedQuadratic()
sim_module = SimQuadratic()
eval_module = EvalQuadratic(lstsq)
jitted_grad_function = jax.jit(
    gradfunction(embedding_module, sim_module, eval_module, sim_aux_data)
)
design_search = GradientDescent(eval_module, jitted_grad_function)

for epoch in range(epochs):
    embedding = embedding_module(design)
    state = sim_module(embedding, sim_aux_data)
    val = eval_module(state, points)
    grads = jitted_grad_function(design, points)
    design = design_search.search(design, points)

    # print objective and gradient sum
    if epoch % 100 == 0:
        print(val, sum(grads))
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < 0.01:
        print(epoch, jnp.sum(grads), design)
        break

xnew = jnp.linspace(sim_aux_data[0], sim_aux_data[-1], 1000)
plt.plot([p.x for p in points], [p.y for p in points], "bo")
final_embedding = embedding_module(design)
plt.plot(xnew, sim_module(embedding_module(design), xnew))
plt.show()

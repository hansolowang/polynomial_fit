from api import *

from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class EmbedQuadraticFunction(DesignEmbedding):
    def __call__(self, x):
        return lambda y: jnp.polyval(x, y)


class SimQuadraticFunction(DesignSimulation):
    def __call__(self, embedding, horizion):
        return embedding(horizon)


class EvalQuadraticFunction(DesignEvaluation):
    def grad(self, x, objectives):
        def f(design):
            state = jnp.polyval(design, horizon)
            loss = 0.0
            for obj in objectives:
                loss += (state[obj.x] - obj.y) ** 2
            return loss
        return jax.grad(f)(design)


class Objective(eqx.Module):
    x: int = eqx.field(static=True)
    y: float

obj0 = Objective(0, 1.0)
obj1 = Objective(1, -20.0)
obj2 = Objective(2, 10.0)
obj3 = Objective(3, -.5)
obj4 = Objective(4, -10.0)
# objectives = [obj0, obj1, obj2, obj3, obj4]
objectives = [obj0, obj1]


design = jnp.zeros(3, dtype='float32')
horizon = np.linspace(0, 5, 6)
lr = 1e-3
epochs = 500

design_search = DesignSearch()

for epoch in range(epochs):
    f = EmbedQuadraticFunction().embed(design)
    state = SimQuadraticFunction().simulate(f, horizon)
    val = EvalQuadraticFunction().val(state, objectives)
    grads = EvalQuadraticFunction().grad(design, objectives)
    design = design_search.search(design, grads, lr)

    # print objective and gradient sum
    if epoch % 100 == 0:
        print(val, sum(grads))
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < .01:
        print(epoch, jnp.sum(grads), design)
        break

xnew = jnp.linspace(horizon[0], horizon[-1], 1000)
plt.plot([o.x for o in objectives], [o.y for o in objectives], 'bo')
plt.plot(xnew, jnp.polyval(design, xnew))
plt.show()

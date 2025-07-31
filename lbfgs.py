from dataclasses import dataclass
import equinox as eqx
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

"""
val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
design = DesignSearch(design, val, grad)
"""
class Objective(eqx.Module):
    x: int = eqx.field(static=True)
    y: float

class DesignConstraints(eqx.Module):
    x: int = eqx.field(static=True)
    upper_bound: float
    lower_bound: float

class StateConstraints(eqx.Module):
    x: int = eqx.field(static=True)
    upper_bound: float
    lower_bound: float

obj0 = Objective(0, 1.0)
obj1 = Objective(1, -20.0)
obj2 = Objective(2, 10.0)
obj3 = Objective(3, -.5)
obj4 = Objective(4, -10.0)
# some sort of notion of what the environment. Use a function to generate the objectives.
objectives = [obj0, obj1, obj2, obj3, obj4]

def DesignSearch(design, grads, lr):
    """Improve low dimension representation"""
    design = design - lr * grads
    return design


def DesignEmbedding(design):
    return lambda x: jnp.polyval(design, x)


def DesignSimulation(design_embedding, x):
    """
    Reconstruct low dimension representation to high level representation
    """
    state = design_embedding(x)
    return state


def ObjectiveFunction(state, objs: list[Objective]):
    """passed into DesignEvaluation"""
    loss = 0
    for obj in objs:
        loss += (state[obj.x] - obj.y) ** 2
    return loss


@jax.jit
def DesignEvaluation(state, objs):
    loss_val = ObjectiveFunction(state, objs)
    return loss_val

@jax.jit
def GradDesignEvaluation(design, horizon, objs):
    def f(design):
        state = jnp.polyval(design, horizon)
        loss = 0.0
        for obj in objs:
            loss += (state[obj.x] - obj.y) ** 2
        return loss
    return jax.grad(f)(design)


design = jnp.zeros(4, dtype='float32')
lr = 1e-6
epochs = 5000

# make this a function
for epoch in range(epochs):
    horizon = np.linspace(0, 5, 6)
    obj_loss = DesignEvaluation(DesignSimulation(DesignEmbedding(design), horizon), objectives)
    grads = GradDesignEvaluation(design, horizon, objectives)
    design = DesignSearch(design, grads, lr)

    # print objective and gradient sum
    if epoch % 100 == 0:
        print(obj_loss, sum(grads))
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < .01:
        print(epoch, jnp.sum(grads), design)
        break

xnew = jnp.linspace(horizon[0], horizon[-1], 1000)
plt.plot([o.x for o in objectives], [o.y for o in objectives], 'bo')
plt.plot(xnew, DesignSimulation(DesignEmbedding(design), xnew))
plt.show()

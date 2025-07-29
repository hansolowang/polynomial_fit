from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Objective:
    x: float
    y: float


obj0 = Objective(0, 1.0)
obj1 = Objective(1, -20.0)
obj2 = Objective(2, 10.0)
obj3 = Objective(3, -.5)
obj4 = Objective(4, -10.0)
# some sort of notion of what the environment. Use a function to generate the objectives.
objectives = [obj0, obj1, obj2, obj3, obj4]

@dataclass
class Constraint:
    x: float # NO CLUE YET
    

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
        loss += (state[obj.x] - y) ** 2
    return loss

# def ObjectiveFunction(state, y):
#     return jnp.mean((state - y) ** 2)

def ConstraintFunction():
    """constraint on embedding result or simulation result"""


def DesignEvaluation2(objective_function, state, y):
    return objective_function(state, y)

def DesignEvaluation(state, objectives, design):
    """Calculate the values of objectives and constraints"""
    def loss_fn(state, y):
        return jnp.mean((state - y)**2)
    obj_loss = loss_fn(state, y)
    # Hmmmm
    def loss_fn_for_gradient(design, x, y):
        preds = jnp.polyval(design, x)
        return jnp.mean((preds - y)**2)
    grads = jax.grad(loss_fn_for_gradient)(design, x, y)

    return obj_loss, grads

design = jnp.zeros(4)
lr = 5e-4
epochs = 50000


# make this a function
for epoch in range(epochs):
    """
    val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
    grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
    design = DesignSearch(design, val, grad)
    """
    horizon = np.linspace(0, 100, 101)
    state = DesignSimulation(DesignEmbedding(design), horizon)
    obj_loss, grads = DesignEvaluation(state, x, y, design)
    design = DesignSearch(design, grads, lr)

    # print objective and gradient sum
    if epoch % 100 == 0:
        print(obj_loss, sum(grads))
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < .01:
        print(epoch, jnp.sum(grads), design)
        break

xnew = jnp.linspace(x[0], x[-1], 1000)
plt.plot(x, y, 'bo')
plt.plot(xnew, DesignSimulation(DesignEmbedding(design), xnew))
plt.show()

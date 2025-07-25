import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# some sort of notion of what the environment. Use a function to generate the objectives.
x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = jnp.array([1.0, -20.0, 10., -5., -10.])

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

def ObjectiveFunction(state, y):
    """passed into DesignEvaluation"""
    pass

def ObjectiveFunction(state, y):
    return jnp.mean((state - y) ** 2)

def DesignEvaluation2(objective_function, state, y):
    return objective_function(state, y)

def DesignEvaluation(state, x, y, design):
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
lr = 1e-4
epochs = 100000


# make this a function
for epoch in range(epochs):
    """
    val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
    grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
    design = DesignSearch(design, val, grad)
    """
    state = DesignSimulation(DesignEmbedding(design), x)
    obj_loss, grads = DesignEvaluation(state, x, y, design)
    design = DesignSearch(design, grads, lr)

    # print 
    if epoch % 100 == 0:
        print(obj_loss, sum(grads), obj_loss2)
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < .01:
        print(epoch)
        print(jnp.sum(grads), design)
        break

xnew = jnp.linspace(x[0], x[-1], 1000)
plt.plot(x, y, 'bo')
plt.plot(xnew, DesignSimulation(design, xnew))
plt.show()

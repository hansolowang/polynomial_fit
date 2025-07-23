import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = jnp.array([1.0, -20.0, 10., -5., -10.])

def DesignSearch(design, grads, lr):
    """Improve low dimension representation"""
    design = design - lr * grads
    return design

def DesignSimulation(design, x):
    """
    Reconstruct low dimension representation to high level representation
    """
    return jnp.polyval(design, x)

def DesignEvaluation(design, x, y):
    """Calculate the values of objectives and constraints"""
    def loss_fn(design, x, y):
        preds = jnp.polyval(design, x)
        return jnp.mean((preds - y)**2)
    val = loss_fn(design, x, y)
    grads = jax.grad(loss_fn)(design, x, y)
    return val, grads

design = jnp.zeros(4)
lr = 1e-4
epochs = 100000
for epoch in range(epochs):
    val, grads = DesignEvaluation(design, x, y)
    design = DesignSearch(design, grads, lr)
    if epoch % 100 == 0:
        print(jnp.sum(grads), design)
    # break if nan or gradient is trivial
    if any(jnp.isnan(grads)) or abs(jnp.sum(grads)) < .01:
        print(epoch)
        print(jnp.sum(grads), design)
        break


xnew = jnp.linspace(x[0], x[-1], 1000)
plt.plot(x, y, 'bo')
plt.plot(xnew, DesignSimulation(design, xnew))
plt.show()

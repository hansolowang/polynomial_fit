from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from enum import Enum

@dataclass
class Objectives:
    x: np.ndarray
    y: np.ndarray

def DesignSearch(design: np.ndarray, gradient_vector: np.ndarray=None) -> np.ndarray:
    """Improve low dimension representation"""
    design += 0.0001 * gradient_vector
    return design

def DesignSimulation(design: np.ndarray):
    """
    Reconstruct low dimension representation to high level representation
    """
    def f(x):
        return sum(x**idx*val for idx, val in enumerate(design))
    return f
#     return np.polynomial.polynomial.Polynomial(design)

def DesignEvaluation(model: np.ndarray, obj: Objectives, constraints=None) ->np.ndarray:
    """Calculate the values of objectives and constraints"""
    # plot
    loss = 0
    for x,y in zip(obj.x, obj.y):
        loss += (model(x) - y)**2
    return loss

#    xnew = np.linspace(obj.x[0], obj.x[-1], 1000)
#    plt.plot(obj.x, obj.y, 'bo')
#    plt.plot(xnew, model(xnew))
#    plt.show()

def DesignGradient(model, objectives):
    return np.array([grad_a, grad_b, grad_c, grad_d])

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, -20.0, 10., -5., -10.])
objs = Objectives(x, y)

design=np.array([0.0, 0.0, 0.0, 0.0])
grads = np.array([0.0, 0.0, 0.0, 0.0])

for i in range(100):
    design = DesignSearch(design, grads)
    model = DesignSimulation(design)
    loss = DesignEvaluation(model, objs)
    grads = DesignGradient(model, objs)
    print(f"loss:{loss} grads:{grads} grad_norm: {sum(grads)}")
DesignEvaluation(model, objs)

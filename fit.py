from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from enum import Enum

"""
Gradient based
Evolutionary based
"""

# class Method(Enum):
#     Gradient = 1
#     Genetic = 2
#     Oneshot = 3



@dataclass
class Objectives:
    x: np.ndarray
    y: np.ndarray

def DesignSearch(design: np.ndarray, fitness_vector: np.ndarray=None, objectives: Objectives=None, constraints=None) -> np.ndarray:
    """Improve low dimension representation"""
    # Oneshot
    popt = Polynomial.fit(objectives.x, objectives.y, design.shape[0])
    design = popt.convert(domain=[0,1], window=[0,1]).coef
    return design

def DesignSimulation(design: np.ndarray):
    """
    Reconstruct low dimension representation to high level representation
    """
    def f(x):
        return sum(x**idx*val for idx, val in enumerate(design))
    return f

def DesignEvaluation(model: np.ndarray, obj: Objectives, constraints=None) ->np.ndarray:
    """Calculate the values of objectives and constraints"""
    # plot
    xnew = np.linspace(obj.x[0], obj.x[-1], 1000)
    plt.plot(obj.x, obj.y, 'bo')
    plt.plot(xnew, model(xnew))
    plt.show()



x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, -20.0, 10., -5., -10.])
objs = Objectives(x, y)

design=np.array([0, 0, 0, 0])

design = DesignSearch(design, objectives=objs)
model = DesignSimulation(design)
DesignEvaluation(model, objs)

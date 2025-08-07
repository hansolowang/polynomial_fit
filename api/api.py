from datacasses import dataclass
from abc import ABC, abstractmethod
import jax

"""
val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
design = DesignSearch(design, val, grad)
"""


class DesignEmbedding(ABC):
    @abstractmethod
    def __call__(self, x):
        ...

    # Maybe a seperate method instead of __call__
    # def generate():


class DesignSimulation(ABC):
    @abstractmethod
    def __call__(self, embedding, sim_aux_data):
        ...

    # Maybe a seperate instead of __call__
    # def simulate():


class DesignEvaluation(ABC):
    def __init__(self, objectives):
        self.objectives = objectives

    @abstractmethod
    def __call__(self, state, eval_aux_data):
        ...


class DesignSearch(ABC):
    def __init__(self, value_function, grad_function):
        self.value_function = value_function
        self.gradient_function = grad_function

    @abstractmethod
    def search(self, x, search_aux_data):
        ...


# TODO: does horizon make sense as an argument here?
def gradfunction(design_embedding, design_simulation, design_evaluation, sim_aux_data):
    def f(design, eval_aux_data):
        return design_evaluation(
            design_simulation(design_embedding(design), sim_aux_data), eval_aux_data
        )

    return jax.grad(f)

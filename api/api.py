from abc import ABC, abstractmethod

"""
val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
design = DesignSearch(design, val, grad)
"""


# DesignVector = np.ndarray
# RobotSystem = "high level representation"
# SimulationStates = "time varying simulation state"
# RobotSystemConstraint = "physical constraint data structure"
# SimulationStateConstraint = "simulation state constraint data structure"
class DesignEmbedding(ABC):
    @abstractmethod
    def __call__(self, x):
        ...


class DesignSimulation(ABC):
    @abstractmethod
    def __call__(self, embedding, horizon):
        ...


class DesignEvaluation(ABC):
    def __init__(self, objectives):
        self.objectives = objectives

    def val(self, state):
        loss = 0
        for obj in self.objectives:
            loss += (state[obj.x] - obj.y) ** 2
        return loss

    def grad(self, x, objectives):
        raise NotImplementedError


class DesignSearch(ABC):
    def search(self, x, grads, lr):
        return x - lr * grads

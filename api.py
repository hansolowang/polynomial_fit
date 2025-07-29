def 

"""
val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
design = DesignSearch(design, val, grad)
"""

DesignVector = np.ndarray
RobotSystem = "high level representation"
SimulationStates = "time varying simulation state"
RobotSystemConstraint = "physical constraint data structure"
SimulationStateConstraint = "simulation state constraint data structure"



def DesignEmbedding(DesignVector):
    return RobotSystem

def DesignSimulation(RobotSystem):
    return SimulationStates

def DesignObjective(RobotSystem or SimulationStates):
    return np.ndarray

def DesignConstraint():
    pass

def DesignEvaluation():
    pass

def DesignSearch():
    pass


# make this a function
for epoch in range(epochs):
    val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))), DesignConstraint(DesignSimulation(DesignEmbedding(design))))
    grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))), DesignConstraint(DesignSimulation(DesignEmbedding(design))))
    design = DesignSearch(design, val, grad)

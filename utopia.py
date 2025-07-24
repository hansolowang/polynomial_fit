def 

"""
val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))
grad = Grad(DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design)))))
design = DesignSearch(design, val, grad)
"""


def DesignEmbedding():
    pass

def DesignSimulation():
    pass

def DesignObjective():
    pass

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

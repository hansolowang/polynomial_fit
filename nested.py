from dataclasses import dataclass
import equinox as eqx
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

val = DesignEvaluation(DesignObjective(DesignSimulation(DesignEmbedding(design))))

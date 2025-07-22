from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit


test = Polynomial([1,2,3])
print(test)

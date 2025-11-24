
import numpy as np
from dynamics.ab_algorithm import ABAlgorithm

np.set_printoptions(precision=2, suppress=True)

ab = ABAlgorithm()
print(ab.jcalc('Rz', 1.57))
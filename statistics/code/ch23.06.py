import numpy as np

from numpy import linalg


P = np.array([
    [0.4, 0.5, 0.1],
    [0.05, 0.7, 0.25],
    [0.05, 0.5, 0.45]
])

el, ev = linalg.eig(P.T)
print(1 / el[0] * ev[:, 0])

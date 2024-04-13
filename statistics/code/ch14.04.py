# I make it myself very easy.

import numpy as np

rng = np.random.RandomState(37)
nsim = 10

mean = np.array([3, 8])
cov = np.array([[1, 1], [1, 2]])

print(rng.multivariate_normal(mean, cov, size=nsim))

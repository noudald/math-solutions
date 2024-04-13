import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, t


def f(x):
    return norm.pdf(x)


def g(x):
    return 1 / (1 + x**2)


n = 1000
rng = np.random.RandomState(37)

y = []
for _ in range(n):
    x = rng.standard_cauchy()
    u = rng.uniform(0, 1)
    while u > f(x)/g(x):
        x = rng.standard_cauchy()
        u = rng.uniform(0, 1)
    y.append(x)

plt.hist(y)
plt.show()

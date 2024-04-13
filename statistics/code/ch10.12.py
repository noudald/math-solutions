import numpy as np

from scipy.stats import norm

l_0 = 1.0
n = 20
alpha = 0.05

rng = np.random.RandomState(37)

n_simulations = 10**5
x = rng.poisson(l_0, size=(n_simulations, n))
l_hat = x.mean(axis=1)
wald = np.sqrt(n) * (l_hat - l_0) / np.sqrt(l_0)

n_rejections = np.sum(np.abs(wald) > norm.ppf(1 - alpha / 2))
print(n_rejections / n_simulations)

import numpy as np

from scipy.stats import norm


n1 = 200
n2 = 200
x1 = 160
x2 = 148

p1_hat = x1/n1
p2_hat = x2/n2
psi_hat = p1_hat - p2_hat


# Delta method
se_hat = np.sqrt(p1_hat*(1 - p1_hat)/n1 + p2_hat*(1 - p2_hat)/n2)

print('Delta method 90% confidence interval: ({:.2f}, {:.2f})'.format(
    psi_hat - norm.ppf(.95)*se_hat, psi_hat + norm.ppf(.95)*se_hat
))


# Parametric bootstrapping.
bs_nsim = 10**4
bs_rng = np.random.RandomState(37)
bs_x1 = bs_rng.binomial(n1, p1_hat, size=bs_nsim)
bs_x2 = bs_rng.binomial(n2, p2_hat, size=bs_nsim)
bs_psi = bs_x1/n1 - bs_x2/n2

print('Parametric bootstrapping interval: ({:.2f}, {:.2f})'.format(
    np.quantile(bs_psi, .05), np.quantile(bs_psi, .95)
))

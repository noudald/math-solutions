import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


mu = 5
n = 100
rng = np.random.RandomState(37)
x = rng.normal(mu, 1, size=n)

# Part (a)

# Delta method.
mu_hat = x.mean()
print(mu_hat)
se_hat = np.exp(mu_hat) / np.sqrt(n)
print('95% confidence interval Delta method:             ({:.2f}, {:.2f})'.format(
    np.exp(mu_hat) - norm.ppf(0.95)*se_hat,
    np.exp(mu_hat) + norm.ppf(0.95)*se_hat
))

# Parametric bootstrap
pbs_nsim = 10**4
pbs_x = rng.normal(mu_hat, 1, size=(n, pbs_nsim))
pbs_mu = np.mean(pbs_x, axis=0)
pbs_t = np.exp(pbs_mu)
print('95% confidence interval parametric bootstrap:     ({:.2f}, {:.2f})'.format(
    np.quantile(pbs_t, 0.025), np.quantile(pbs_t, 0.975)
))

# Non-parametric bootstrap
npbs_nsim = 10**4
npbs_x = rng.choice(x, size=(n, npbs_nsim), replace=True)
npbs_mu = np.mean(npbs_x, axis=0)
npbs_t = np.exp(npbs_mu)
print('95% confidence interval non-parametric bootstrap: ({:.2f}, {:.2f})'.format(
    np.quantile(npbs_t, 0.025), np.quantile(npbs_t, 0.975)
))


# Part (b)

t_true = np.exp(np.mean(rng.normal(mu, 1, size=(n, 10**4)), axis=0))
t_delta = rng.normal(np.exp(mu_hat), se_hat, size=10**4)

plt.plot(np.sort(t_true), np.linspace(0, 1, len(t_true), endpoint=False), label='true')
plt.plot(np.sort(t_delta), np.linspace(0, 1, len(t_delta), endpoint=False), label='delta')
plt.plot(np.sort(pbs_t), np.linspace(0, 1, len(pbs_t), endpoint=False), label='parametric bootstrap')
plt.plot(np.sort(npbs_t), np.linspace(0, 1, len(npbs_t), endpoint=False), label='non-parametric bootstrap')

plt.legend()

plt.show()

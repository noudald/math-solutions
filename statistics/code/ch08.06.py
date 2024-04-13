import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm


n = 100
mu = 5
n_simulations = 10**4
rng = np.random.RandomState(37)

x = rng.normal(mu, 1, size=(n,))
theta = np.exp(mu)
theta_hat = np.exp(x.mean())

print(f'Real theta: {theta:.2f}')

# (a) Use bootstrap to get se and the 95% percentage confidence interval of theta.
bootstrap_theta = np.zeros(n_simulations)
for i in range(n_simulations):
    bootstrap_x = rng.choice(x, size=x.size, replace=True)
    bootstrap_theta_hat = np.exp(bootstrap_x.mean())
    bootstrap_theta[i] = bootstrap_theta_hat

theta_error = bootstrap_theta.std()

print(f'Estimated theta: {theta_hat:.2f} +/- {theta_error:.3f}')
print('Estimated 95% confidence interval (normal): [{:.3f}, {:.3f}]'.format(
    theta_hat - norm.ppf(0.975)*theta_error,
    theta_hat + norm.ppf(0.975)*theta_error
))
print('Estimated 95% confidence interval (pivotal): [{:.3f}, {:.3f}]'.format(
    2*theta_hat - np.quantile(bootstrap_theta, 0.975),
    2*theta_hat - np.quantile(bootstrap_theta, 0.025)
))
print('Estimated 95% confidence interval (percentile): [{:.3f}, {:.3f}]'.format(
    np.quantile(bootstrap_theta, 0.025),
    np.quantile(bootstrap_theta, 0.975)
))


# (b) Plot the histogram of the bootstrap.
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

t = np.linspace(100, 220, 100)
true_dist = np.sqrt(n)/t * norm.pdf(np.sqrt(n)*(np.log(t) - mu))

ax.plot(t, true_dist)
ax.hist(bootstrap_theta, bins=50, density=True)

fig.show()
plt.show()

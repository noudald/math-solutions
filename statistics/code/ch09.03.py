import numpy as np

from scipy.stats import norm

data = np.array([
    3.23, -2.50,  1.88, -0.68,  4.43, 0.17,
    1.03, -0.07, -0.01,  0.76,  1.76, 3.18,
    0.33, -0.31,  0.30, -0.61,  1.52, 5.43,
    1.54,  2.28,  0.42,  2.33, -1.03, 4.00,
    0.39
])

mu_hat = data.mean()
sigma_hat = data.std()
tau_hat = norm.ppf(0.95)*sigma_hat + mu_hat

print(f'tau_hat: {tau_hat:.2f}')

alpha = 0.05

# Delta method.
se_hat = sigma_hat*np.sqrt((2 + norm.ppf(0.95)**2)/(2*data.shape[0]))

print('Estimate tau, Delta method 95% confidence interval: ({:.2f}, {:.2f})'.format(
    tau_hat - norm.ppf(1 - alpha/2)*se_hat,
    tau_hat + norm.ppf(1 - alpha/2)*se_hat
))

# Parametric bootstrapping.
rng = np.random.RandomState(37)
bs_nsim = 10**5
bs_nsamples = data.shape[0]
bs_taus = np.quantile(
    rng.normal(mu_hat, sigma_hat, size=(bs_nsamples, bs_nsim)),
    0.95,
    axis=0
)

print('Estimate tau, parametric bootstrap, 95% confidence interval: ({:.2f}, {:.2f})'.format(
    np.quantile(bs_taus, 0.025),
    np.quantile(bs_taus, 0.975)
))

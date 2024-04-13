import numpy as np

from scipy.stats import norm


n = 50
n_bootstrap = 10**5
alpha = 0.05

rng = np.random.RandomState(37)

Y = rng.normal(0, 1, size=(n,))
X = np.exp(Y)

def skewness(samples):
    mu_hat = samples.mean()
    sigma_hat = samples.std()
    return np.mean((samples - mu_hat)**3 / sigma_hat**3)

bootstrap_skewness = np.zeros(n_bootstrap)
for i in range(n_bootstrap):
    bootstrap_X = rng.choice(X, size=n, replace=True)
    bootstrap_skewness[i] = skewness(bootstrap_X)

print('Skewness 95% confidence interval bootstrap (normal):     [{:>5.2f}, {:>5.2f}]'.format(
    bootstrap_skewness.mean() - norm.ppf(1 - alpha/2)*bootstrap_skewness.std(),
    bootstrap_skewness.mean() + norm.ppf(1 - alpha/2)*bootstrap_skewness.std()
))
print('Skewness 95% confidence interval bootstrap (pivot):      [{:>5.2f}, {:>5.2f}]'.format(
    2*bootstrap_skewness.mean() - np.quantile(bootstrap_skewness, 1 - alpha/2),
    2*bootstrap_skewness.mean() - np.quantile(bootstrap_skewness, alpha/2),
))
print('Skewness 95% confidence interval bootstrap (percentile): [{:>5.2f}, {:>5.2f}]'.format(
    np.quantile(bootstrap_skewness, alpha/2), np.quantile(bootstrap_skewness, 1 - alpha/2)
))

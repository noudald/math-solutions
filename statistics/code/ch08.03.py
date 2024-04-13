import numpy as np

from scipy.stats import norm


rng = np.random.RandomState(37)
n = 25
n_simulations = 10**4
alpha = 0.05


def bootstrap_func(samples):
    return (np.quantile(samples, .75) - np.quantile(samples, .25)) / 1.34


X = rng.standard_t(3, size=(n,))
bootstrap_result = np.zeros(n_simulations)
for i in range(n_simulations):
    bootstrap_samples = rng.choice(X, size=n, replace=True)
    bootstrap_result[i] = bootstrap_func(bootstrap_samples)

print('Bootstrap 95% interval (normal):     [{:.2f}, {:.2f}]'.format(
    bootstrap_result.mean() - bootstrap_result.std()*norm.ppf(1 - alpha/2),
    bootstrap_result.mean() + bootstrap_result.std()*norm.ppf(1 - alpha/2)
))
print('Bootstrap 95% interval (pivot):      [{:.2f}, {:.2f}]'.format(
    2*bootstrap_result.mean() - np.quantile(bootstrap_result, 1 - alpha/2),
    2*bootstrap_result.mean() - np.quantile(bootstrap_result, alpha/2)
))
print('Bootstrap 95% interval (percentile): [{:.2f}, {:.2f}]'.format(
    np.quantile(bootstrap_result, alpha/2),
    np.quantile(bootstrap_result, 1 - alpha/2)
))

import numpy as np

from scipy.stats import norm
from tqdm import tqdm


lsat = np.array([576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575,
                 545, 572, 594])
gpa = np.array([3.39, 3.30, 2.81, 3.03, 3.44, 3.07, 3.00, 3.43, 3.36, 3.13,
                3.12, 2.74, 2.76, 2.88, 3.96])


def correlation(x, y):
    return (np.sum((x - x.mean())*(y - y.mean()))
        / np.sqrt(np.sum((x - x.mean())**2) * np.sum((y - y.mean())**2)))


theta_hat = correlation(lsat, gpa)

n_simulations = 10**4
rng = np.random.RandomState(37)
bootstrap_correlation = np.zeros(n_simulations)
for i in range(n_simulations):
    bootstrap_lsat = rng.choice(lsat, size=lsat.size, replace=True)
    bootstrap_gpa = rng.choice(gpa, size=gpa.size, replace=True)
    bootstrap_correlation[i] = correlation(bootstrap_lsat, bootstrap_gpa)

theta_error = bootstrap_correlation.std()

print(f'Estimated correlation: {theta_hat:.2f} +/- {theta_error:.3f}')
print('Estimated 95% confidence interval (normal): [{:.3f}, {:.3f}]'.format(
    theta_hat - norm.ppf(0.975)*theta_error,
    theta_hat + norm.ppf(0.975)*theta_error
))
print('Estimated 95% confidence interval (pivotal): [{:.3f}, {:.3f}]'.format(
    2*theta_hat - np.quantile(bootstrap_correlation, 0.975),
    2*theta_hat - np.quantile(bootstrap_correlation, 0.025)
))
print('Estimated 95% confidence interval (percentile): [{:.3f}, {:.3f}]'.format(
    np.quantile(bootstrap_correlation, 0.025),
    np.quantile(bootstrap_correlation, 0.975)
))

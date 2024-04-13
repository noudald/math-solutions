import numpy as np
from scipy.stats import norm, binom_test, chisquare, chi2


data = np.array([
    [55, 141],
    [33, 145],
    [70, 139],
    [49, 161]
])

# Test 1, Wald per week on fractions.
print('Test 1, Wald test per week with BH corrected p-values')
total = data.sum(axis=0)
fc = data[:, 0] / total[0]
fj = data[:, 1] / total[1]
f_hat = fc - fj
se_hat = np.sqrt(fc * (1 - fc) / total[0] + fj * (1 - fj) / total[1])

wald = f_hat / se_hat
p_values = 2 * norm.cdf(-abs(wald))

# Benjamini-Hochberg correction
alpha = 0.05
step = 1 / data.shape[0]
r = np.argwhere(np.sort(p_values) < alpha * np.arange(step, 1.0 + step, step))[-1][0]
threshold = np.sort(p_values)[r]

for w, p in zip([-2, -1, 1, 2], p_values):
    print(f'Week {w:>2} p-value {p:.4f} reject {p <= threshold}')


# Test 2, Binomial test
print('\nTest 2, binomial test per week with BH corrected p-values')
theta_hat = data[:, 0].sum() / data.sum()
p_values = np.array([
    binom_test(data[i, 0], data[i, :].sum(), theta_hat)
       for i in range(len(data))
])

# Benjamini-Hochberg correction
alpha = 0.05
step = 1 / data.shape[0]
r = np.argwhere(np.sort(p_values) < alpha * np.arange(step, 1.0 + step, step))[-1][0]
threshold = np.sort(p_values)[r]

for w, p in zip([-2, -1, 1, 2], p_values):
    print(f'Week {w:>2} p-value {p:.4f} reject {p <= threshold}')


# Test 3, X^2-test
print('\nTest 3, X^2-test')
theta_0 = data[:, 0].sum() / data.sum()
chi2_ = ((data[:, 0] - theta_0 * data.sum(axis=1))**2 / (theta_0 * data.sum(axis=1))).sum()
df = len(data) - 1
print(f'p-value: {1 - chi2.cdf(chi2_, df):.5f}')

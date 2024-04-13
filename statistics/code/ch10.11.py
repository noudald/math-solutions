import numpy as np

from scipy.stats import norm

num_patients = np.array([80, 75, 85, 67, 85])
num_nausea = np.array([45, 26, 52, 35, 37])
num_normal = num_patients - num_nausea

p_hat = num_nausea / num_patients
se_hat = np.sqrt(p_hat[0] * (1 - p_hat[0]) / num_patients[0] + p_hat[1:] * (1 - p_hat[1:]) / num_patients[1:])
wald = (p_hat[0] - p_hat[1:]) / se_hat
p_values = 2 * norm.cdf(-np.abs(wald))

print('p-values', p_values)

odd_ratio = (p_hat[1:] / (1 - p_hat[1:])) / (p_hat[0] / (1 - p_hat[0]))
print('odds ratio', odd_ratio)

# Benjamini-Hochberg correction
alpha = 0.05
step = 1 / len(p_values)
r = np.argwhere(np.sort(p_values) < alpha * np.arange(step, 1.0 + step, step))[-1][0]
threshold = np.sort(p_values)[r]

for p in p_values:
    print(f'{p:>5.3f}', p <= threshold)

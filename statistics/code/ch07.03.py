import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from tqdm import tqdm

rng = np.random.RandomState(37)
n_observations = 100
alpha = 0.05
epsilon = np.sqrt(1 / (2*n_observations) * np.log(2 / alpha))

x_range = np.linspace(-3, 3, 100)

X = rng.normal(0, 1, size=n_observations)
X.sort()
F_hat = np.array([sum(X <= x) / n_observations for x in x_range])
L = np.maximum(F_hat - epsilon, 0)
U = np.minimum(F_hat + epsilon, 1)

fig, ax = plt.subplots(2, 1, figsize=(12, 2*8))

ax[0].plot(x_range, norm.cdf(x_range), label='True distribution')
ax[0].plot(x_range, F_hat, label='$\hat{F}$')
ax[0].plot(x_range, U, label='Lower bound')
ax[0].plot(x_range, L, label='Upper bound')
ax[0].set(title='Normal')
ax[0].legend()


X = rng.standard_cauchy(size=n_observations)
X.sort()
F_hat = np.array([sum(X <= x) / n_observations for x in x_range])
L = np.maximum(F_hat - epsilon, 0)
U = np.minimum(F_hat + epsilon, 1)

ax[1].plot(x_range, norm.cdf(x_range), label='True distribution')
ax[1].plot(x_range, F_hat, label='$\hat{F}$')
ax[1].plot(x_range, U, label='Lower bound')
ax[1].plot(x_range, L, label='Upper bound')
ax[1].set(title='Chaucy')
ax[1].legend()

plt.show()


n_experiments = 1000

outside_bounds = 0
for _ in tqdm(range(n_experiments)):
    X = rng.normal(0, 1, size=n_observations)
    X.sort()
    F_hat = np.array([sum(X <= x) / n_observations for x in x_range])
    L = np.maximum(F_hat - epsilon, 0)
    U = np.minimum(F_hat + epsilon, 1)

    if np.any(np.logical_or(norm.cdf(x_range) < L, U < norm.cdf(x_range))):
        outside_bounds += 1

print(f'Emperical distribution function for normal distribution was within bounds: {1 - outside_bounds/n_experiments:.2f}')


outside_bounds = 0
for _ in tqdm(range(n_experiments)):
    X = rng.standard_cauchy(size=n_observations)
    X.sort()
    F_hat = np.array([sum(X <= x) / n_observations for x in x_range])
    L = np.maximum(F_hat - epsilon, 0)
    U = np.minimum(F_hat + epsilon, 1)

    if np.any(np.logical_or(norm.cdf(x_range) < L, U < norm.cdf(x_range))):
        outside_bounds += 1

print(f'Emperical distribution function for Cauchy distribution was within bounds: {1 - outside_bounds/n_experiments:.2f}')


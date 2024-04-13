import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

fig, ax = plt.subplots(2, 1, figsize=(12, 2*8))

# 4.4.(b)
rng = np.random.RandomState(37)
n_simulations = 10**4

n = np.arange(1, 10001, 1)
coverage = []
for n_ in tqdm(n):
    a = 0.05
    p = 0.4
    e = np.sqrt(1/(2*n_)*np.log(2/a))
    X = rng.binomial(n_, p, size=(n_simulations,))
    p_hat = X / n_

    coverage.append(np.mean((p_hat - e < p) & (p < p_hat + e)))

ax[0].set(xlabel='n', ylabel='coverage')
ax[0].plot(n, coverage)


# 4.4.(c)
e = np.sqrt(1/(2*n)*np.log(2/a))

ax[1].plot(n, 2*e)
ax[1].set(xlabel='n', ylabel='$|C_p|$')

plt.show()

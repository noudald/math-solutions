import numpy as np

import matplotlib.pyplot as plt


rng = np.random.RandomState(37)

n = 5
n_simulations = 10**4

fig, ax = plt.subplots(4, 1, figsize=(6, 4*4))

for i, n in enumerate([1, 5, 25, 100]):
    X_hat = np.zeros(n_simulations)
    for j in range(n_simulations):
        X = rng.random_sample(size=(n,))
        X_hat[j] = np.mean(X)

    ax[i].hist(X_hat)
    ax[i].set(xlim=[0, 1])

fig.show()
plt.show()

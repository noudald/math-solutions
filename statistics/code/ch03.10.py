import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)

n_samples = 10000
n_simulations = 10

for _ in range(n_simulations):
    simulation = np.cumsum(1 - 2*rng.randint(0, 2, size=(n_samples,)))
    plt.plot(np.arange(1, n_samples + 1), simulation)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)
n = 1000

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

n_range = np.arange(1, n + 1)

p = .3
coin = [rng.binomial(k, p) for k in n_range]
ax.plot(n_range, coin, color='blue', label='p = {}'.format(p))
ax.plot(n_range, p * n_range, '--', color='red')

ax.set(xlabel='n', ylabel='$np$')
ax.legend()

fig.show()
plt.show()

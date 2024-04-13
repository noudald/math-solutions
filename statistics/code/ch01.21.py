import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)
n = 1000

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

n_range = np.arange(1, n + 1)

p = .3
coin = rng.binomial(1, p, size=(n,))
p_hat = [sum(coin[:k] == 1) / k for k in n_range]
ax.plot([1, n], [p, p], '--', color='blue')
ax.plot(n_range, p_hat, color='blue', label='p = {}'.format(p))

p = .03
coin = rng.binomial(1, p, size=(n,))
p_hat = [sum(coin[:k] == 1) / k for k in n_range]
ax.plot([1, n], [p, p], '--', color='red')
ax.plot(n_range, p_hat, color='red', label='p = {}'.format(p))

ax.set(xlabel='n', ylabel='$\hat{p}$')
ax.legend()

fig.show()
plt.show()

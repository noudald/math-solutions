import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(37)
n = 50
theta = 1

x = rng.uniform(0, theta, size=n)

bs_n = 10**5
bs_max = np.zeros(bs_n)
for i in range(bs_n):
    bs_sample = rng.choice(x, size=x.shape[0], replace=True)
    bs_max[i] = np.max(bs_sample)

plt.hist(bs_max, bins=100)
plt.vlines(theta, 0, bs_n, color='red')
plt.show()

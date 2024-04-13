import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(37)


# 17.8.(b)
def approx_pyz(nsim):
    samples = rng.uniform(0, 1, size=(nsim, 3))

    x = (samples[:,0] > 0.5).astype(int)
    y = (samples[:,1] > 1 / (1 + np.exp(4*x - 2))).astype(int)
    z = (samples[:,2] > 1 / (1 + np.exp(2*x + 2*y - 2))).astype(int)

    return np.sum((y == 1) & (z == 1)) / np.sum(y == 1)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

nsims = np.linspace(0, 10**6, 100)
ax.scatter(nsims, [approx_pyz(int(nsim)) for nsim in nsims])
true_prop = 0.5 * 1 / (1 + np.exp(2)) + (np.exp(2) / (1 + np.exp(2)))**2
ax.hlines(true_prop, nsims.min(), nsims.max(), linestyle='--')

fig.show()
plt.show()


# 17.8.(d)
def approx_p1z(nsim):
    samples = rng.uniform(0, 1, size=(nsim, 2))

    x = (samples[:,0] > 0.5).astype(int)
    y = np.ones(nsim)
    z = (samples[:,1] > 1 / (1 + np.exp(2*x + 2*y - 2))).astype(int)

    return np.sum((y == 1) & (z == 1)) / np.sum(y == 1)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

nsims = np.linspace(0, 2*10**5, 100)
ax.scatter(nsims, [approx_p1z(int(nsim)) for nsim in nsims])
true_prop = 0.25 + 0.5 * np.exp(2) / (1 + np.exp(2))
ax.hlines(true_prop, nsims.min(), nsims.max(), linestyle='--')

fig.show()
plt.show()

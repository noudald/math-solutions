import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 1, figsize=(8, 2*6))

# Part (a)
y = np.linspace(0.01, 5, 100)
fy = 1 / (2 * np.pi) * np.exp(-1/2 * np.log(y))

ax[0].plot(y, fy)
ax[0].set(title='Probability density function', xlim=[0, 5], ylim=[0, 1])

# Part (b)
rng = np.random.RandomState(37)
X = rng.normal(0, 1, size=(10000,))
Y = np.exp(X)

ax[1].hist(Y, bins=50)
ax[1].set(title='Simulation histogram', xlim=[0, 5])

fig.show()
plt.show()

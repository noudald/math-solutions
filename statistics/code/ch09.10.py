import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)
n = 50
theta = 1

x = rng.uniform(0, theta, size=n)
x_max_hat = np.max(x)

# Non-Parametric bootstrap
npbs_n = 10**4
npbs_x = rng.choice(x, size=(n, npbs_n), replace=True)
npbs_x_max = np.max(npbs_x, axis=0)

# Parametric bootstrap
pbs_n = 10**4
pbs_x = rng.uniform(0, x_max_hat, size=(n, pbs_n))
pbs_x_max = np.max(pbs_x, axis=0)

plt.plot([0, 0.99999, 1], [0, 0, 1], label='true')
plt.plot(np.sort(npbs_x_max), np.linspace(0, 1, len(npbs_x_max), endpoint=False), label='non-parametric bootstrap')
plt.plot(np.sort(pbs_x_max), np.linspace(0, 1, len(pbs_x_max), endpoint=False), label='parametric bootstrap')
plt.show()

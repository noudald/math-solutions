import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)

sample_mean_normal = []
sample_mean_cauchy = []
sample_n = np.arange(1, 10001)

for n in sample_n:
    X_normal = rng.normal(0, 1, size=(n,))
    X_cauchy = rng.standard_cauchy(size=(n,))

    sample_mean_normal.append(np.mean(X_normal))
    sample_mean_cauchy.append(np.mean(X_cauchy))

fig, ax = plt.subplots(2, 1, figsize=(12, 2*8))

ax[0].plot(sample_n, sample_mean_normal)
ax[0].set(title='Normal sample mean')
ax[1].plot(sample_n, sample_mean_cauchy)
ax[1].set(title='Cauchy sample mean')

fig.show()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import beta


x = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
n = x.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ls = np.linspace(0, 1, 500)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, (a, b) in enumerate([(0.5, 0.5), (1.0, 1.0), (10.0, 10.0), (100.0, 100.0)]):
    ax.plot(
        ls,
        beta.pdf(ls, a, b),
        '--',
        color=colors[i],
        label='$\mathrm{Beta}' + f'({a:.1f}, {b:.1f})$'
    )
    ax.plot(
        ls,
        beta.pdf(ls, x.sum() + a, n - x.sum() + b),
        color=colors[i],
        label='$p \sim \mathrm{Beta}' + f'({a:.1f}, {b:.1f})$'
    )
ax.legend()

fig.show()
plt.show()

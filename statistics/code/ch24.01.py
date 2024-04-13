import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


def h(x):
    return np.exp(-x**2/2) / np.sqrt(2 * np.pi)


# Solution (a)

n = 10000
rng = np.random.RandomState(37)
sample_x = rng.uniform(1, 2, size=n)
sample_h = h(sample_x)
I = norm.cdf(2) - norm.cdf(1)
I_hat = sample_h.mean()
I_std = sample_h.std() / np.sqrt(n)

print('Solution (a)')
print(f'I ~= {I_hat:.6f} +/- {2*I_std:.6f}')


# Solution (b)

I_ana = np.sqrt(1 / (n - 1) * (I - I**2))
print('\nSolution (b)')
print(f'I =  {I:.6f} +/- {2*I_ana:.6f}')


# Solution (c), I'm not entirely sure what they are asking me to do.

def importance_sampling(v, n, n_bs=10000):
    sample_x = rng.normal(1.5, v, size=(n, n_bs))
    sample_h = ((1 <= sample_x) & (sample_x <= 2)).astype(float)
    sample_f = norm(0, 1).pdf(sample_x)
    sample_g = norm(1.5, v).pdf(sample_x)
    sample_hfg = sample_h * sample_f / sample_g

    I_hat = np.mean(sample_hfg, axis=0)

    return I_hat.mean(), sample_hfg.std(), sample_hfg.mean(axis=0)

print('\nSolution (c)')
fig, ax = plt.subplots(3, 1, figsize=(8, 3*4))
for i, v in enumerate((0.1, 1.0, 10.0)):
    I_hat, I_std, I_samples = importance_sampling(v, n)
    ax[i].hist(I_samples, bins=100)
    ax[i].set(title=f'v = {v}')
    print(f'I({v}) ~= {I_hat:.6f} +/- {2*I_std:.6f}')

fig.show()
plt.show()

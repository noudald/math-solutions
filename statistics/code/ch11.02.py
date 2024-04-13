import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


rng = np.random.RandomState(37)
n = 100
mu = 5
x = rng.normal(mu, 1, size=n)

# (b)
ls = np.linspace(4.5, 5.5, 100)
mu_pdf = norm.pdf(ls, x.mean(), 1/np.sqrt(n))

# (c): Not 100% sure if I understand the question. I think the pdf should be
# compared with the bootstrap values.
ndraw = 1000
x_draw = rng.choice(x, size=(ndraw, n))
mu_draw = np.mean(x_draw, axis=1)


# (d)
exp_mu_draw = np.exp(mu_draw)
exp_mu_ls = np.linspace(50, 250, 100)
exp_mu_pdf = np.sqrt(n)/exp_mu_ls * norm.pdf(np.sqrt(n)*(np.log(exp_mu_ls) - x.mean()))


fig, ax = plt.subplots(2, 1, figsize=(8, 2*4))

ax[0].plot(ls, mu_pdf)
ax[0].hist(mu_draw, bins=100, density=True)

ax[1].plot(exp_mu_ls, exp_mu_pdf)
ax[1].hist(exp_mu_draw, bins=100, density=True)

fig.show()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.special import gamma

t = np.linspace(0.1, 1.0, 100)

norm_cdf = 2*norm.cdf(-t)

markov_1 = np.sqrt(2 / np.pi) * gamma((1 + 1)/2) / t
markov_2 = np.sqrt(2**2 / np.pi) * gamma((2 + 1)/2) / t
markov_3 = np.sqrt(2**3 / np.pi) * gamma((3 + 1)/2) / t
markov_4 = np.sqrt(2**4 / np.pi) * gamma((4 + 1)/2) / t
markov_5 = np.sqrt(2**5 / np.pi) * gamma((5 + 1)/2) / t

mills = np.sqrt(2 / np.pi) * 1/t * np.exp(-t**2/2)

plt.plot(t, norm_cdf, label='$P(|Z| > t)')
plt.plot(t, markov_1, label='$E(|Z|)/t$')
plt.plot(t, markov_2, label='$E(|Z|^2)/t^2$')
plt.plot(t, markov_3, label='$E(|Z|^3)/t^3$')
plt.plot(t, markov_4, label='$E(|Z|^4)/t^4$')
plt.plot(t, markov_5, label='$E(|Z|^5)/t^5$')
plt.plot(t, mills, label='Mills')

plt.legend()

plt.show()

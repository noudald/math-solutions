import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma


rng = np.random.RandomState(37)
t1 = 1.5
t2 = 2.0
n = 1000


# Solution (a), independence-Metropolis-Hastings method.

def f(z):
    return z**(-3/2) * np.exp(-t1*z - t2/z + 2*np.sqrt(t1*t2) + np.log(np.sqrt(2*t2)))

a = 1.2

def g(x):
    return gamma.pdf(x, a=a)

y = rng.gamma(a, size=n)
x = [1.0]
for i in range(1, n):
    r = min(1, (f(y[i - 1]) * g(x[i - 1])) / (f(x[i - 1]) * g(y[i - 1])))
    if rng.uniform(0, 1) < r:
        x.append(y[i - 1])
    else:
        x.append(x[-1])

print(np.mean(x[0:]), np.sqrt(t2 / t1))

plt.plot(x)
plt.show()


# Solution (b), random-walk-Metropolis-Hasting method.

def f(z):
    return -3/2 * np.log(z) - t1*z - t2/z + 2*np.sqrt(t1*t2) + np.log(np.sqrt(2*t2))

y = rng.normal(size=n)
x = [1.0]
for i in range(1, n):
    r = min(1, f(y[i - 1]) / f(x[i - 1]))
    if rng.uniform(0, 1) < r:
        x.append(y[i - 1])
    else:
        x.append(x[-1])

print(np.mean(np.exp(x[0:])), np.sqrt(t2 / t1))

plt.plot(np.exp(x[1:]))
plt.show()

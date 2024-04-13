'''
I'm not so sure if the results are correct. I followed everything from the book
to the letter, but the results seem a bit off. In particular the confidence
intervals.
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm


with open('./data/glass.dat', 'r') as f:
    columns = f.readline().split()
    data = {key: [] for key in columns}

    for line in f.readlines():
        for key, value in zip(columns, line.split()[1:]):
            data[key].append(value)

df = pd.DataFrame(data)
X = df['Al'].values.astype(float)
Y = df['RI'].values.astype(float)


def kernel(x):
    if isinstance(x, float):
        return kernel(np.array([x]))[0]
    else:
        r = 0.75 * (1 - 0.2*x**2) / np.sqrt(5)
        r[np.abs(x) > np.sqrt(5)] = 0
        return r

def r(x, h, s, y):
    '''Nadaraya-Watson kernel estimator.'''
    if isinstance(x, float):
        return r(np.array([x]), h, s, y)[0]
    else:
        ki = kernel((x.reshape(-1, 1) - s.reshape(1, -1))/h)
        wi = ki / np.sum(ki, axis=1).reshape(-1, 1)
        wi[np.isnan(wi)] = 0.0
        ri = np.sum(wi * y.reshape(1, -1), axis=1)

        return ri

def r_conf(x, h, s, y, alpha=0.05):
    m = 2*np.sqrt(5)/h
    q = norm.ppf(0.5*(1 + (1 - alpha)**(1/m)))

    y_sorted = np.sort(y)
    n = len(y_sorted)
    s2 = 1/(2*(n - 1))*np.sum(np.diff(y_sorted, 1))

    ki = kernel((x.reshape(-1, 1) - s.reshape(1, -1))/h)
    wi = ki / np.sum(ki, axis=1).reshape(-1, 1)
    wi[np.isnan(wi)] = 0.0

    ri = np.sum(wi * y.reshape(1, -1), axis=1)

    se = np.sqrt(s2 * np.sum(wi**2, axis=1))

    ri_lb = ri - q*se
    ri_ub = ri + q*se

    return ri, ri_lb, ri_ub


def Jh(h, s, y):
    # TODO: Optimize with numpy.
    n = len(y)
    idx = np.arange(0, n)
    jh = []
    for h_ in h:
        dsum = 0.0
        for i in idx:
            loo = idx[idx != i]
            yi_hat = r(s[i], h_, s[loo], y[loo])
            dsum += (y[i] - yi_hat)**2
        jh.append(dsum)

    return jh


h_space = np.linspace(0.01, 1.0, 100)
jh = Jh(h_space, X, Y)
h_opt = h_space[np.argmin(jh)]

x_range = np.linspace(X.min(), X.max(), 100)
ri, ri_lb, ri_ub = r_conf(x_range, h_opt, X, Y)

fig, ax = plt.subplots(1, 2, figsize=(2*6, 4))

ax[0].plot(h_space, jh)

ax[1].plot(x_range, ri, color='red')
ax[1].plot(x_range, ri_lb, linestyle='--', color='red')
ax[1].plot(x_range, ri_ub, linestyle='--', color='red')
ax[1].scatter(X, Y, marker='+')
ax[1].set(xlabel='Al', ylabel='RI')

fig.show()
plt.show()

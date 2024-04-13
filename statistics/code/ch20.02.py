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

X = df['RI'].astype(float).values


# (a) Histogram Density Estimation
def bars(m, x):
    xmin, xmax = x.min(), x.max()
    h = (xmax - xmin)/m

    return [(xmin + i*h, xmin + (i+1)*h) for i in range(m)]

def fhat(m, x):
    xmin, xmax = x.min(), x.max()
    h = (xmax - xmin)/m
    n = len(x)

    phat = []
    for i in range(m - 1):
        phat.append(((xmin + i*h <= x) & (x < xmin + (i + 1)*h)).sum() / n)
    phat.append(1 - sum(phat))

    return np.array(phat)

def J_hist(m, x):
    xmin, xmax = x.min(), x.max()
    h = (xmax - xmin)/m
    n = len(x)

    p2sum = (fhat(m, x)**2).sum()

    # Why do a need a +? This could be a mistake in the book.
    return 2*m/(n - 1) + (n + 1)/(n - 1)*p2sum


mrange = list(range(1, 101))
Jm = [J_hist(m, X) for m in mrange]
m_opt = mrange[np.argmin(Jm)]

f_hat = fhat(m_opt, X)
f_bar = bars(m_opt, X)

n = len(X)
alpha = 0.05
c = norm.ppf(1 - alpha/(2*m_opt)) / 2 * np.sqrt(m_opt/n)
ln = np.sqrt(f_hat) - c
ln[ln < 0] = 0
ln = ln**2
un = (np.sqrt(f_hat) + c)**2

fig, ax = plt.subplots(2, 2, figsize=(2*6, 2*4))

ax[0, 0].plot(mrange, Jm)
ax[0, 0].set(title='Histogram Density $\hat{J}(h)$')
ax[0, 0].set(xlabel='Number of bins')

ax[0, 1].bar(list(map(lambda x: (x[1] + x[0])/2, f_bar)), f_hat, color='blue')
ax[0, 1].bar(list(map(lambda x: (x[1] + x[0])/2, f_bar)), un, color='black', alpha=0.2)
ax[0, 1].bar(list(map(lambda x: (x[1] + x[0])/2, f_bar)), ln, color='red')
ax[0, 1].set(title='Histogram Density Estimation')


# (b) Kernel Density Estimation
def kernel(x):
    if isinstance(x, float):
        return kernel(np.array([x]))[0]
    else:
        r = 0.75 * (1 - 0.2*x**2) / np.sqrt(5)
        r[np.abs(x) > np.sqrt(5)] = 0
        return r

def kernel2(x):
    if isinstance(x, float):
        return kernel2(np.array([x]))[0]
    else:
        ys = np.linspace(-np.sqrt(5), np.sqrt(5), 10)
        xy = x.reshape(-1, 1) - ys

        kernel_xy = kernel(xy)
        kernel_y = kernel(ys)

        kernel2 = 2*np.sqrt(5)/len(ys) * np.sum(kernel_xy * kernel_y, axis=1)

        return kernel2

def kde(x, s, h):
    m = len(x)
    n = len(s)
    return 1/n * 1/h * np.sum(kernel((x.reshape(m, -1) - s.reshape(-1, n))/h), axis=1)

def kde_conf(x, s, h, alpha=0.05):
    n = len(s)
    m = 2*np.sqrt(5)*h
    q = norm.ppf(0.5*(1 + (1 - alpha)**(1/m)))
    yi = 1/h * kernel(1/h*(x.reshape(-1, 1) - s.reshape(1, -1)))
    s2 = 1/(n - 1) * np.sum((yi - yi.mean(axis=1).reshape(-1, 1))**2, axis=1)
    se = np.sqrt(s2/n)

    fn = kde(x, s, h)
    fn_lb = fn - q*se
    fn_lb[fn_lb < 0] = 0.0
    fn_ub = fn + q*se
    fn_ub[fn_ub > 1] = 1.0

    return fn, fn_lb, fn_ub

def Jkde(s, h):
    n = len(s)
    sstar = 1/h * (s.reshape(-1, 1) - s.reshape(1, -1))
    t1 = 1/(h*n**2) * np.sum(kernel2(sstar) - 2*kernel(sstar).reshape(-1))
    t2 = 2/(h*n) * kernel(0.0)

    return t1 + t2

h_space = np.linspace(0.05, 1.0, 100)
h_jkde = np.array([Jkde(X, h_) for h_ in h_space])
h_opt = h_space[np.argmin(h_jkde)]
ax[1, 0].plot(h_space, h_jkde)
ax[1, 0].set(title='Kernel Density $\hat{J}(h)$')
ax[1, 0].set(xlabel='Kernel width')

x_space = np.linspace(X.min() - 1, X.max() + 1, 500)
fn, fn_lb, fn_up = kde_conf(x_space, X, h_opt)
ax[1, 1].plot(x_space, fn, color='blue')
ax[1, 1].plot(x_space, fn_lb, linestyle='--', linewidth=1, color='black', alpha=0.5)
ax[1, 1].plot(x_space, fn_up, linestyle='--', linewidth=1, color='black', alpha=0.5)
ax[1, 1].set(title='Kernel Density Estimation')

fig.tight_layout()
fig.show()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import chi


with open('./data/glass.dat', 'r') as f:
    columns = f.readline().split()
    data = {key: [] for key in columns}

    for line in f.readlines():
        for key, value in zip(columns, line.split()[1:]):
            data[key].append(value)

df = pd.DataFrame(data).sort_values(by='Al')
X = df['Al'].values.astype(float)

# Scale values such that they are between 0 and 1
X = (X - X.min())/(X.max() - X.min())

Y = df['RI'].values.astype(float)


# Part (a)

def basis(j, x):
    if j == 0:
        return np.ones(x.shape)
    else:
        return np.sqrt(2)*np.cos(j*np.pi*x)

def bj_hat(j, x, y, N=1000):
    if j == 0:
        return np.mean(y)
    else:
        return np.mean(y * basis(j, x))

def f_hat(x, n, X, Y, alpha=0.05):
    bj = np.array([bj_hat(j, X, Y) for j in range(0, n+1)])
    basisj = np.array([basis(j, x) for j in range(0, n+1)])

    # Confidence band
    k = n // 4
    s2 = n/k * np.sum([bj[i]**2 for i in range(n - k + 1, n + 1)])
    a = np.sqrt(np.sum(np.array([basis(j, x)**2 for j in range(1, n + 1)]), axis=0))

    f_hat_ = np.sum(bj.reshape(-1, 1) * basisj, axis=0)
    lx = f_hat_ - a * (s2**.5) * chi.ppf(1 - alpha, n) / np.sqrt(n)
    ux = f_hat_ + a * (s2**.5) * chi.ppf(1 - alpha, n) / np.sqrt(n)

    return f_hat_, lx, ux

def osre(X, Y, n):
    '''Find optimal number of basis elements for estimation.'''
    k = n // 4
    bj = np.array([bj_hat(j, X, Y) for j in range(0, n+1)])
    s2 = n/k * np.sum([bj[i]**2 for i in range(n - k + 1, n + 1)])

    j_min, r_min = 0, 10**10
    for j in range(0, n + 1):
        scoef = np.array([bj[i]**2 - s2/n for i in range(j + 1, n + 1)])
        scoef[scoef < 0] = 0
        r_hat = j*s2/n + np.sum(scoef)
        if r_hat < r_min:
            j_min = j
            r_min = r_hat

    return j_min, r_min


n_opt, _ = osre(X, Y, 100)
x_space = np.linspace(0, 1, 1000)
f_hat_, lx, ux = f_hat(x_space, n_opt, X, Y)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X, Y)
ax.plot(x_space, f_hat_, color='black')
ax.plot(x_space, lx, color='black', linestyle='--')
ax.plot(x_space, ux, color='black', linestyle='--')

fig.show()
plt.show()


# Part (b)

def haar(x, j, k):
    def haar_(x):
        y = np.zeros(x.shape)
        y[(0 <= x) & (x <= 0.5)] = -1
        y[(0.5 < x) & (x <= 1.0)] = 1
        return y

    return 2**(j/2) * haar_(2**j*x - k)

def hwr(x, X, Y, n_max):
    a_hat = np.mean(Y)
    djk = np.array([[np.mean(haar(X, j, k)*Y) for k in range(0, 2**j)] for j in range(0, n_max)])
    s = np.sqrt(n_max) / 0.6745 * np.abs(np.median(djk[-1]))
    t = s * np.sqrt(2 * np.log(n_max) / n_max)

    f_hat = a_hat * np.ones(x.shape)
    for j in range(0, n_max):
        for k in range(0, 2**j):
            if np.abs(djk[j][k]) > t:
                f_hat += djk[j][k] * haar(x, j, k)

    return f_hat

x_space = np.linspace(0, 1, 1000)
f_hat_3 = hwr(x_space, X, Y, 3)
f_hat_5 = hwr(x_space, X, Y, 5)
f_hat_8 = hwr(x_space, X, Y, 8)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X, Y, s=4, color='black')
ax.plot(x_space, f_hat_3, label='J = 3', linewidth=1)
ax.plot(x_space, f_hat_5, label='J = 5', linewidth=1)
ax.plot(x_space, f_hat_8, label='J = 8', linewidth=1)
ax.legend()

fig.show()
plt.show()

# plt.plot(x_space, haar(x_space, 2, 2))
# plt.show()

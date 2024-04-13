import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


rng = np.random.RandomState(37)
n = 1000

X = rng.beta(15, 4, size=(n,))
x = np.linspace(X.min(), X.max(), 100)


def haar(x, j, k):
    def haar_(x):
        y = np.zeros(x.shape)
        y[(0 <= x) & (x <= 0.5)] = -1
        y[(0.5 < x) & (x <= 1.0)] = 1
        return y

    return 2**(j/2) * haar_(2**j*x - k)


def haar_histogram(x, X, b):
    a_hat = np.mean(X)
    bjk = [
        [np.mean(haar(X, j, k)) for k in range(0, 2**j)]
        for j in range(0, b + 1)
    ]

    f_hat = a_hat * np.ones(x.shape)
    for j in range(0, b + 1):
        for k in range(0, 2**j):
            f_hat += bjk[j][k] * haar(x, j, k)

    return f_hat


def J_hat(X, b):
    x = np.linspace(X.min(), X.max(), 100)
    n = len(X)

    fsum = np.mean(haar_histogram(x, X, b))

    loo = 0
    for i in range(n):
        X_loo = X[np.arange(n) != i]
        loo += haar_histogram(np.array([X[i]]), X_loo, b)[0]

    return fsum - 2/n * loo


search_space = range(10)
loo = [J_hat(X, b) for b in tqdm(search_space)]

fig, ax = plt.subplots(2, 1, figsize=(8, 2*4))

ax[0].plot(search_space, loo)

ax[1].hist(X, density=True)
ax[1].plot(x, haar_histogram(x, X, np.argmin(loo)))

fig.show()
plt.show()

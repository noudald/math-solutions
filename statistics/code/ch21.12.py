import numpy as np
import matplotlib.pyplot as plt


def haar(x, j, k):
    def haar_(x):
        y = np.zeros(x.shape)
        y[(0 <= x) & (x <= 0.5)] = -1
        y[(0.5 < x) & (x <= 1.0)] = 1
        return y

    return 2**(j/2) * haar_(2**j*x - k)


def basis(j, k, x):
    if j == 0:
        return np.ones(x.shape)
    else:
        return haar(x, j, k)


def bj_hat(j, k, f_func, N=1000):
    int_space = np.linspace(0, 1, N)
    if j == 0:
        return np.mean(f_func(int_space))
    else:
        return np.mean(f_func(int_space) * basis(j, k, int_space))


def f_hat(x, n, f_func):
    params = [(j, k) for j in range(0, n + 1) for k in range(0, 2**j)]
    bj = np.array([bj_hat(j, k, f_func) for j, k in params])
    basisj = np.array([basis(j, k, x) for j, k in params])

    return np.sum(bj.reshape(-1, 1) * basisj, axis=0)


def plot_approximation(f_func):
    fig, ax = plt.subplots(4, 1, figsize=(6, 4*4))

    x_range = np.linspace(0, 1, 200)
    ax[0].plot(x_range, f_func(x_range), color='black')

    ax[1].plot(x_range, f_hat(x_range, 3, f_func), alpha=0.5, label=f'n=3')
    ax[1].legend()

    ax[2].plot(x_range, f_hat(x_range, 5, f_func), alpha=0.5, label=f'n=4')
    ax[2].legend()

    ax[3].plot(x_range, f_hat(x_range, 8, f_func), alpha=0.5, label=f'n=8')
    ax[3].legend()

    fig.show()
    plt.show()


# (a)
def fa(x):
    return np.sqrt(2) * np.cos(3 * np.pi * x)

plot_approximation(fa)


# (b)
def fb(x):
    return np.sin(np.pi * x)

plot_approximation(fb)


# (c)
def kernel(t):
    return (1 + np.sign(t)) / 2

def fc(x):
    tj = np.array([.1, .13, .15, .23, .24, .40, .44, .65, .76, .78, .81])
    hj = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    return np.sum(hj.reshape(-1, 1)*kernel(x.reshape(1, -1) - tj.reshape(-1, 1)), axis=0)

plot_approximation(fc)


# (d)
def fd(x):
    return np.sqrt(x*(1 - x)) * np.sin(2.1*np.pi/(x + .05))

plot_approximation(fd)

import numpy as np
import matplotlib.pyplot as plt


def basis(j, x):
    if j == 0:
        return np.ones(x.shape)
    else:
        return np.sqrt(2)*np.cos(j*np.pi*x)

def bj_hat(j, f_func, N=1000):
    int_space = np.linspace(0, 1, N)
    if j == 0:
        return np.mean(f_func(int_space))
    else:
        return np.mean(f_func(int_space) * basis(j, int_space))

def f_hat(x, n, f_func):
    bj = np.array([bj_hat(j, f_func) for j in range(0, n+1)])
    basisj = np.array([basis(j, x) for j in range(0, n+1)])
    return np.sum(bj.reshape(-1, 1) * basisj, axis=0)

def plot_approximation(f_func):
    fig, ax = plt.subplots(2, 1, figsize=(6, 2*4))

    x_range = np.linspace(0, 1, 1000)
    ax[0].plot(x_range, f_func(x_range), color='black')
    for n in [2, 10, 100, 1000]:
        ax[1].plot(x_range, f_hat(x_range, n, f_func), alpha=0.5, label=f'n={n}')
    ax[1].legend()

    fig.show()
    plt.show()


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

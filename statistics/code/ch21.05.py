import numpy as np
import matplotlib.pyplot as plt

def Legendre(i, x):
    if not isinstance(x, np.ndarray):
        return Legendre(i, np.array([x]))
    if i == 0:
        return np.ones(x.shape)
    elif i == 1:
        return x
    else:
        return ((2*i - 1)*x*Legendre(i - 1, x) - (i - 1)*Legendre(i - 2, x)) / i


n = 100
x = np.linspace(-1, 1, n)

for i in range(5):
    Li = Legendre(i, x)
    for j in range(5):
        Lj = Legendre(j, x)
        print(f'<L_{i}, L_{j}> = {np.dot(Li, Lj)/n:>5.2f}')

fig, ax = plt.subplots(5, 1, figsize=(3, 5*4))

for i in range(5):
    ax[i].plot(x, [Legendre(i, x_) for x_ in x])

fig.show()
plt.show()

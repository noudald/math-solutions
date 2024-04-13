import numpy as np

from scipy.special import gamma


def median(d, n):
    return ((1 - 0.5**(1/n)) / (np.pi**(1/d) / gamma(d/2 + 1)))**(1/d)


n = 100
d = 1
while True:
    if median(d, n) > 0.5:
        print(f'n = {n}, d = {d}')
        break
    d += 1


n = 1000
d = 1
while True:
    if median(d, n) > 0.5:
        print(f'n = {n}, d = {d}')
        break
    d += 1


n = 10000
d = 1
while True:
    if median(d, n) > 0.5:
        print(f'n = {n}, d = {d}')
        break
    d += 1

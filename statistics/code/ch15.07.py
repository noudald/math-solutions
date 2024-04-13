import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm


with open('./data/calcium.dat', 'r') as f:
    for _ in range(33):
        f.readline()

    columns = f.readline().split()
    data = {key: [] for key in columns}

    for line in f.readlines():
        data[columns[0]].append(line.split()[0])
        for key, value in zip(columns[1:], line.split()[1:]):
            data[key].append(int(value))

df = pd.DataFrame(data)


# Two sample Kolmogorov-Smirnov test.
x1 = df[df['Treatment'] == 'Calcium']['Decrease'].values
x2 = df[df['Treatment'] == 'Placebo']['Decrease'].values
y = np.sort(np.concatenate([x1, x2]))

f1 = np.array([sum(x1 < y_)/len(x1) for y_ in y])
f2 = np.array([sum(x2 < y_)/len(x2) for y_ in y])
D = np.max(np.abs(f1 - f2))

def kolmogorov(t):
    return 1 - 2*np.sum([(-1)**(j-1)*np.exp(-2*j**2*t**2) for j in range(1, 10)])

n1 = x1.shape[0]
n2 = x2.shape[0]
print('Two sample Kolmogorov-Smirnov test statistic {:.2f} (p-value: {:.2f})'.format(
    D,
    1 - kolmogorov(np.sqrt(n1*n2/(n1 + n2))*D))
)

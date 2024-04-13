import numpy as np
import pandas as pd

from scipy.stats import chi2


with open('./data/montana.dat', 'r') as f:
    for _ in range(31):
        f.readline()

    columns = f.readline().split()
    data = {key: [] for key in columns}

    f.readline()

    for line in f.readlines():
        for key, value in zip(columns, line.split()):
            if value == '*':
                data[key].append(np.nan)
            else:
                data[key].append(int(value))

df = pd.DataFrame(data)

df_selection = df[['AGE', 'FIN']].dropna()

class_age = df_selection['AGE'].unique()
class_fin = df_selection['FIN'].unique()

x = np.zeros(shape=(len(class_age), len(class_fin)), dtype=int)
for i, age in enumerate(class_age):
    for j, fin in enumerate(class_fin):
        x[i, j] = len(df_selection[(df_selection['AGE'] == age) & (df_selection['FIN'] == fin)])

n = x.sum()
xi_ = x.sum(axis=1)
x_j = x.sum(axis=0)
eij = xi_.reshape(3, 1) @ x_j.reshape(1, 3) / n
df = (len(class_age) - 1)*(len(class_fin) - 1)


# Likelihood ratio test statistic.
T = 2 * (x * np.log(x / eij)).sum()

print('Likelihood ratio statistic: {:.2f}'.format(T))
print('Likelihood ratio test p-value: {:.6f}'.format(
    1 - chi2.cdf(T, df)
))


# Pearson X^2 test.
U = ((x - eij)**2/eij).sum()

print('Pearson X^2 statistic: {:.2f}'.format(U))
print('Pearson X^2 test p-value: {:.6f}'.format(
    1 - chi2.cdf(U, df)
))

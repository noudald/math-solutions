import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm


with open('./data/carmileage.dat', 'r') as f:
    for _ in range(27):
        f.readline()

    columns = f.readline().split()[1:]
    data = {key: [] for key in columns}

    f.readline()

    for line in f.readlines():
        data[columns[0]].append(' '.join(line.split()[:-5]))
        for key, value in zip(columns[1:], line.split()[-5:]):
            data[key].append(float(value))

df = pd.DataFrame(data)


# (a) Find regression HP = b0 + b1 * MPG

x = df.HP.values
y = df.MPG.values

n = len(y)

b1 = np.sum((x - x.mean())*(y - y.mean()))/np.sum((x - x.mean())**2)
b0 = y.mean() - b1*x.mean()

lin_a = np.linspace(x.min(), x.max(), 100)
pred_a = b0 + b1*lin_a

sigma2_hat = 1 / (len(y) - 2) * np.sum((y - (b0 + b1*x))**2)

se_b0 = np.sqrt(sigma2_hat * np.mean(x**2) / np.sum((x - x.mean())**2))
se_b1 = np.sqrt(sigma2_hat / np.sum((x - x.mean())**2))

t0 = b0 / se_b0
t1 = b1 / se_b1

p0 = 2 * (1 - norm.cdf(np.abs(t0)))
p1 = 2 * (1 - norm.cdf(np.abs(t1)))

rmse = np.sqrt(1 / (len(y) - 2) * np.sum((y - b0 - b1*x)**2))

print(f'(a) RMSE: {rmse:.2f}')
print(pd.DataFrame({
    'parameter': ['(intercept)', '(mpg)'],
    'value': [b0, b1],
    't-score': [t0, t1],
    'p-value': [p0, p1]
}))
print('')


# (b) Find regression HP = b0 + b1 * log(MPG)

x = df.HP.values
y = np.log(df.MPG.values)

b1 = np.sum((x - x.mean())*(y - y.mean()))/np.sum((x - x.mean())**2)
b0 = y.mean() - b1*x.mean()

lin_b = np.linspace(x.min(), x.max(), 100)
pred_b = b0 + b1*lin_b

sigma2_hat = 1 / (len(y) - 2) * np.sum((y - (b0 + b1*x))**2)

se_b0 = np.sqrt(sigma2_hat * np.mean(x**2) / np.sum((x - x.mean())**2))
se_b1 = np.sqrt(sigma2_hat / np.sum((x - x.mean())**2))

t0 = b0 / se_b0
t1 = b1 / se_b1

p0 = 2 * (1 - norm.cdf(np.abs(t0)))
p1 = 2 * (1 - norm.cdf(np.abs(t1)))

rmse = np.sqrt(1 / (len(y) - 2) * np.sum((np.exp(y) - np.exp(b0 + b1*x))**2))

print(f'(b) RMSE: {rmse:.2f}')
print(pd.DataFrame({
    'parameter': ['(intercept)', '(mpg)'],
    'value': [b0, b1],
    't-score': [t0, t1],
    'p-value': [p0, p1]
}))


# Plot results

fig, ax = plt.subplots(3, 1, figsize=(6, 3*4))

ax[0].scatter(df.HP, df.MPG)
ax[0].set(xlabel='HP', ylabel='MPG')
ax[0].plot(lin_a, pred_a)

ax[1].scatter(df.HP, np.log(df.MPG))
ax[1].set(xlabel='HP', ylabel='log(MPG)')
ax[1].plot(lin_b, pred_b)

ax[2].scatter(df.HP, df.MPG)
ax[2].set(xlabel='HP', ylabel='exp(log(MPG))')
ax[2].plot(lin_b, np.exp(pred_b))

fig.show()
plt.show()

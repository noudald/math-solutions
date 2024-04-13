import pandas as pd
import numpy as np

from scipy.stats import norm


with open('./data/faithful.dat', 'r') as f:
    for _ in range(25):
        next(f)
    columns = f.readline().split()
    data = {key: [] for key in columns}
    for line in f.readlines():
        for key, value in zip(columns, line.split()[1:]):
            data[key].append(float(value))

df = pd.DataFrame(data)

waiting_error = df.waiting.std() / np.sqrt(df.shape[0])
waiting_mean = df.waiting.mean()
waiting_median = df.waiting.median()

print(f'Mean waiting time: {waiting_mean:.2f} +/- {waiting_error:.2f}')
print('Mean waiting time 90% confidence interval: [{:.2f}, {:.2f}]'.format(
    waiting_mean - norm.ppf(.95)*waiting_error,
    waiting_mean + norm.ppf(.95)*waiting_error
))
print(f'Median waiting time: {waiting_median:.2f}')

# Bootstrap the error of the median. (Next chapter)
n_simulations = 10**5
rng = np.random.RandomState(37)

waiting = df.waiting.values
n_length = len(df.waiting)
median_bootstrap = np.zeros(n_simulations)
for i in range(n_simulations):
    x_bootstrap = rng.choice(waiting, size=n_length, replace=True)
    median_bootstrap[i] = np.median(x_bootstrap)

print('Median waiting time 90% confidence interval: [{:.2f}, {:.2f}]'.format(
    np.quantile(median_bootstrap, .05),
    np.quantile(median_bootstrap, .95)
))

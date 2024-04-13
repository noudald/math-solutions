import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm


with open('./data/temp.dat', 'r') as f:
    for _ in range(31):
        f.readline()

    columns = f.readline().split()
    data = {key: [] for key in columns}

    for line in f.readlines():
        data[columns[0]].append(' '.join(line.split()[:-3]))
        for key, value in zip(columns[1:], line.split()[-3:]):
            data[key].append(float(value))

df = pd.DataFrame(data)
x = df['JanTemp'].values
y = df['Lat'].values

x_sigma2 = 1/(x.shape[0] - 1)*((x - x.mean())**2).sum()
y_sigma2 = 1/(y.shape[0] - 1)*((y - y.mean())**2).sum()
rho_hat = ((x - x.mean())*(y - y.mean())).mean()/np.sqrt(x_sigma2*y_sigma2)

# Fisher's confidence interval.
alpha = 0.05
theta_hat = 0.5*(np.log(1 + rho_hat) - np.log(1 - rho_hat))
theta_se = 1/np.sqrt(x.shape[0] - 3)
theta_a = theta_hat - norm.ppf(1 - alpha/2)*theta_se
theta_b = theta_hat + norm.ppf(1 - alpha/2)*theta_se
rho_a = (np.exp(2*theta_a) - 1)/(np.exp(2*theta_a) + 1)
rho_b = (np.exp(2*theta_b) - 1)/(np.exp(2*theta_b) + 1)

# Bootstrap
bs_rng = np.random.RandomState(37)
bs_nsim = 10**4
bs_idx = bs_rng.randint(0, x.shape[0], size=(bs_nsim, x.shape[0]))

bs_x_samples = x[bs_idx]
bs_x_mean = bs_x_samples.mean(axis=1).reshape(-1, 1)
bs_x_std = np.sqrt(1/(x.shape[0] - 1)*((bs_x_samples - bs_x_mean)**2).sum(axis=1))

bs_y_samples = y[bs_idx]
bs_y_mean = bs_y_samples.mean(axis=1).reshape(-1, 1)
bs_y_std = np.sqrt(1/(y.shape[0] - 1)*((bs_y_samples - bs_y_mean)**2).sum(axis=1))

bs_rho = ((bs_x_samples - bs_x_mean)*(bs_y_samples - bs_y_mean)).mean(axis=1)/(bs_x_std*bs_y_std)

print('Correlation coefficient: {:.2f}'.format(rho_hat))
print('Correlation 95% confidence interval (Fisher):    ({:.2f}, {:.2f})'.format(
    rho_a, rho_b
))
print('Correlation 95% confidence interval (Bootstrap): ({:.2f}, {:.2f})'.format(
    np.quantile(bs_rho, 0.025), np.quantile(bs_rho, 0.975)
))

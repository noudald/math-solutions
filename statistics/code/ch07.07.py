import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

alpha = 0.05

with open('./data/fijiquakes.dat', 'r') as f:
    columns = f.readline().split()
    data = {key: [] for key in columns}
    for line in f.readlines():
        for key, value in zip(columns, line.split()):
            data[key].append(float(value))

df = pd.DataFrame(data)

magnitude = df['mag'].values
magnitude.sort()

x_range = np.unique(magnitude)
F_hat = np.array([sum(magnitude <= x) / magnitude.size for x in x_range])
epsilon = np.sqrt(1 / (2*magnitude.size) * np.log(2 / alpha))
F_lower = np.maximum(F_hat - epsilon, 0)
F_upper = np.minimum(F_hat + epsilon, 1)

plt.plot(x_range, F_hat, color='red')
plt.plot(x_range, F_lower, linestyle='--', color='blue', alpha=0.5, label='95% lower bound')
plt.plot(x_range, F_upper, linestyle='--', color='blue', alpha=0.5, label='95% upper bound')
plt.legend()
plt.show()


theta = (F_hat[x_range == 4.9] - F_hat[x_range == 4.3])[0]
mag_error = magnitude.std() / np.sqrt(magnitude.size)
theta_error = np.sqrt(2)*mag_error

print('Confidence interval for F(4.9) - F(4.3) = [{:.2f}, {:.2f}]'.format(
    theta - 2*theta_error,
    theta + 2*theta_error
))

import pandas as pd
import numpy as np

from scipy.stats import norm


with open('./data/clouds.dat', 'r') as f:
    for _ in range(29):
        next(f)
    columns = f.readline().split()
    data = {key: [] for key in columns}
    for line in f.readlines():
        for key, value in zip(columns, line.split()):
            data[key].append(float(value))

df = pd.DataFrame(data)

unseeded_mean = df['Unseeded_Clouds'].mean()
unseeded_error = df['Unseeded_Clouds'].std() / np.sqrt(df.shape[0])
seeded_mean = df['Seeded_Clouds'].mean()
seeded_error = df['Seeded_Clouds'].std() / np.sqrt(df.shape[0])

theta = seeded_mean - unseeded_mean
theta_error = np.sqrt(seeded_error**2 + unseeded_error**2)

print(f'Estimated theta: {theta:.2f} +/- {theta_error:.2f}.')
print('Estimated theta 95% confidence interval: [{:.2f}, {:.2f}].'.format(
    theta - norm.ppf(0.975)*theta_error,
    theta + norm.ppf(0.975)*theta_error
))

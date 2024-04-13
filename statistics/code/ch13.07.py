import itertools

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
df['Intercept'] = np.ones((len(df), 1))

features = ['Intercept', 'VOL', 'HP', 'SP', 'WT']

x = df[features].values
y = df['MPG'].values

beta_hat = np.linalg.inv(x.T @ x) @ x.T @ y
sigma2_hat = 1/(x.shape[0] - x.shape[1]) * np.sum((y - x @ beta_hat)**2)
var_hat = sigma2_hat * np.linalg.inv(x.T @ x)
se_hat = np.sqrt(np.diag(var_hat))
t_values = beta_hat / se_hat
p_values = 2*(1 - norm.cdf(np.abs(t_values)))

df_result = pd.DataFrame({
    'feature': features,
    'beta_hat': beta_hat,
    'se_hat': se_hat,
    't': t_values,
    'p-values': p_values
})

print('Solution 13.7.(a)')
print(df_result)


# Solution 13.7.(b)

data = {
    'method': [],
    'features': [],
    'Mallow Cp': [],
}

def mallow_cp(feature_subset):
    x_ss = df[feature_subset].values
    y_ss = df['MPG'].values

    beta_ss_hat = np.linalg.inv(x_ss.T @ x_ss) @ x_ss.T @ y_ss
    r_tr = np.sum((y_ss - x_ss @ beta_ss_hat)**2)

    return r_tr + 2 * len(feature_subset) * sigma2_hat


# Forward stepwise Mallow's statistic.
feature_selection = []
feature_remaining = features.copy()
mallow_cp_cur = mallow_cp(feature_selection)

while len(feature_remaining) > 0:
    scores = [
        mallow_cp(feature_selection + [feature]) for feature in feature_remaining
    ]
    best_score_index = np.argmin(scores)
    if scores[best_score_index] < mallow_cp_cur:
        mallow_cp_cur = scores[best_score_index]
        feature_selection.append(feature_remaining[best_score_index])
        feature_remaining.remove(feature_remaining[best_score_index])
    else:
        break


data['method'].append('Forward stepwise')
data['features'].append(feature_selection)
data['Mallow Cp'].append(mallow_cp_cur)


# Backward stepwise Mallow's statistic.
feature_selection = features.copy()
mallow_cp_cur = mallow_cp(feature_selection)

while len(feature_selection) > 0:
    scores = [
        mallow_cp([f for f in feature_selection if f != f_drop]) for f_drop in feature_selection
    ]
    best_score_index = np.argmin(scores)
    if scores[best_score_index] < mallow_cp_cur:
        mallow_cp_cur = scores[best_score_index]
        feature_selection.remove(feature_selection[best_score_index])
    else:
        break

data['method'].append('Backward stepwise')
data['features'].append(feature_selection)
data['Mallow Cp'].append(mallow_cp_cur)


# (c) Zheng-Loh Model Selection Method.

def zheng_loh_score(feature_subset):
    x_ss = df[feature_subset].values
    y_ss = df['MPG'].values

    beta_ss_hat = np.linalg.inv(x_ss.T @ x_ss) @ x_ss.T @ y_ss
    r_tr = np.sum((y_ss - x_ss @ beta_ss_hat)**2)

    return r_tr + len(feature_subset) * sigma2_hat * np.log(x_ss.shape[0])

feature_order = df_result.sort_values('p-values')['feature'].values
scores = [
    zheng_loh_score(feature_order[:i]) for i in range(1, len(feature_order) + 1)
]
best_index = np.argmin(scores)
feature_selection = list(feature_order[:best_index+1])

data['method'].append('Zheng-Loch')
data['features'].append(feature_selection)
data['Mallow Cp'].append(scores[best_index])

print('\nSolution 13.7.(b) & (c)')
print(pd.DataFrame(data))


# (d) Test all combinations with Mallow Cp and BIC.
# Minimizing BIC is equivalent to minimizing k * log(n) + n * log(1/n * RSS).

def bic(feature_subset):
    x_ss = df[feature_subset].values
    y_ss = df['MPG'].values

    beta_ss_hat = np.linalg.inv(x_ss.T @ x_ss) @ x_ss.T @ y_ss
    r_tr = np.sum((y_ss - x_ss @ beta_ss_hat)**2)

    n = x_ss.shape[0]
    k = x_ss.shape[1]

    return n * np.log(1/n * r_tr) + k * np.log(n)


data = {
    'features': [],
    'Mallow Cp': [],
    'BIC': [],
}
for k in range(1, len(features) + 1):
    for feature_selection in itertools.combinations(features, k):
        data['features'].append(list(feature_selection))
        data['Mallow Cp'].append(mallow_cp(list(feature_selection)))
        data['BIC'].append(bic(list(feature_selection)))

df_iter_results = pd.DataFrame(data)

print('\nSolution 13.7.(d)')
print(df_iter_results)

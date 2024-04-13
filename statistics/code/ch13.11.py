import pandas as pd
import numpy as np


with open('./data/coris.dat', 'r') as f:
    columns = f.readline().split(',')
    data = {key.strip(): [] for key in columns[1:]}

    f.readline()
    f.readline()
    f.readline()

    for line in f.readlines():
        for key, value in zip(columns[1:], line.split(',')[1:]):
            data[key.strip()].append(float(value.strip()))


df = pd.DataFrame(data)
df.insert(0, 'intercept', 1)

features = df.columns[:-1]
target = df.columns[-1]


def fit(subset):
    x = df[subset].values
    y = df[target].values.reshape(-1)

    beta = np.zeros(len(subset))

    p_logit = x @ beta.T
    p = 1 / (1 + np.exp(-p_logit))

    ll = 1
    max_iterations = 100
    for _ in range(max_iterations):
        z = np.log(p / (1 - p)) + (y - p)/(p * (1 - p))
        w = np.diag(p*(1 - p))
        beta = np.linalg.inv(x.T @ w @ x) @ x.T @ w @ z

        p_logit = x @ beta.T
        p = 1 / (1 + np.exp(-p_logit))

        ll_new = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if np.abs(ll - ll_new) < 1e-7:
            break
        ll = ll_new

    return ll, beta


def aic(subset):
    ll, _ = fit(subset)
    return ll - len(subset)


feature_selection = list(features)
aic_cur = aic(feature_selection)

while len(feature_selection) > 0:
    scores = [
        aic([f for f in feature_selection if f != f_drop]) for f_drop in feature_selection
    ]
    best_score_index = np.argmax(scores)
    if scores[best_score_index] > aic_cur:
        aic_cur = scores[best_score_index]
        feature_selection.remove(feature_selection[best_score_index])
    else:
        break

print('Best features:', feature_selection)
print('AIC:', aic_cur)

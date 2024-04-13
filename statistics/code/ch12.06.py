import numpy as np
import pandas as pd


nsim = 10**4
rng = np.random.RandomState(37)

results = {
    'k': [],
    'L(x)': [],
    'L(js)': []
}
for k in list(range(2, 21)) + list(range(30, 101, 10)) + list(range(200, 1001, 100)):
    theta = rng.uniform(-10, 10, size=k)

    def loss(t, t_hat):
        return np.sum((t - t_hat)**2, axis=1)

    X = rng.normal(theta, size=(nsim, k))
    js_coef = (1 - (k - 2) / np.sum(X**2, axis=1))
    X_js = js_coef[:, None] * X

    results['k'].append(k)
    results['L(x)'].append(loss(theta, X).mean())
    results['L(js)'].append(loss(theta, X_js).mean())

df = pd.DataFrame(results)
df['Diff (%)'] = (df['L(x)'] - df['L(js)']) / df['L(x)'] * 100

print(df)

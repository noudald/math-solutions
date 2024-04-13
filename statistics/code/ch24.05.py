'''
The implementation seems to be correct, but we always have overflows in
L(new_beta)/L(beta). No matter how I tweak this function, I cannot avoid
floating point overflows. The results are therefore not correct.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


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

X = df[features].values
Y = df[target].values


# (a)

rng = np.random.RandomState(37)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def L(beta):
    bx = beta @ X.T
    return (1 - sigmoid(-np.sum(np.where(Y, bx, 0)))) * sigmoid(-np.sum(np.where(1 - Y, bx, 0)))


n = 10000
sigma = 1
beta = np.zeros(X.shape[1])
bs = [beta]
for i in tqdm(range(n)):
    new_beta = beta.copy()
    for j in range(len(new_beta)):
        Zj = rng.normal(0, sigma)
        new_beta[j] = Zj
        r = min(1, L(new_beta) / L(beta))
        if rng.uniform(0, 1) < r:
            beta = new_beta
        bs.append(list(beta))

bs = np.array(bs)


fig, ax = plt.subplots(5, 2, figsize=(12, 5*6))

for i in range(5):
    ax[i][0].hist(bs[:,2*i], density=True, bins=100)
    ax[i][1].hist(bs[:,2*i + 1], density=True, bins=100)

fig.show()
plt.show()


# (b)

beta = np.zeros(X.shape[1])

p_logit = X @ beta.T
p = 1 / (1 + np.exp(-p_logit))

ll = 1
max_iterations = 100
for _ in range(max_iterations):
    z = np.log(p / (1 - p)) + (Y - p)/(p * (1 - p))
    w = np.diag(p*(1 - p))
    beta = np.linalg.inv(X.T @ w @ X) @ X.T @ w @ z

    p_logit = X @ beta.T
    p = 1 / (1 + np.exp(-p_logit))

    ll_new = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    if np.abs(ll - ll_new) < 1e-7:
        break
    ll = ll_new


print(beta)

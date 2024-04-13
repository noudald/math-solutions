import numpy as np


true_prop = {
    '[0 0 0]': 1/2 * 2/3 * 1/2,
    '[0 0 1]': 1/2 * 2/3 * 1/2,
    '[0 0 2]': 1/2 * 2/3 * 0,
    '[0 1 0]': 1/2 * 1/3 * 0,
    '[0 1 1]': 1/2 * 1/3 * 1/2,
    '[0 1 2]': 1/2 * 1/3 * 1/2,
    '[1 0 0]': 1/2 * 1/3 * 1/2,
    '[1 0 1]': 1/2 * 1/3 * 1/2,
    '[1 1 0]': 1/2 * 2/3 * 0,
    '[1 1 1]': 1/2 * 2/3 * 1/2,
    '[1 1 2]': 1/2 * 2/3 * 1/2
}


rng = np.random.RandomState(37)

n = 1000

hidden = rng.uniform(0, 1, size=(n, 3))

X = (hidden[:,0] < 0.5).astype(int)
Y = np.where(
    X == 0,
    (hidden[:,1] < 1/3).astype(int),
    (hidden[:,1] < 2/3).astype(int)
)
Z = (Y + 2*hidden[:,2]).astype(int)

results = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], axis=1)

bs_nsim = 10**4
bs_counts = np.zeros((bs_nsim, len(true_prop)))
for i in range(bs_nsim):
    bs_idx = rng.choice(range(n), size=n, replace=True)
    bs_results = results[bs_idx, :]
    for j, key in enumerate(true_prop.keys()):
        x = int(key[1])
        y = int(key[3])
        z = int(key[5])
        c = np.sum((bs_results[:,0] == x) & (bs_results[:,1] == y) & (bs_results[:,2] == z))
        bs_counts[i, j] = c

for j, key in enumerate(true_prop.keys()):
    print('{} true: {:.4f} pred: [{:.4f}, {:.4f}]'.format(
        key,
        true_prop[key],
        np.quantile(bs_counts[:,j], 0.025) / n,
        np.quantile(bs_counts[:,j], 0.925) / n,
    ))

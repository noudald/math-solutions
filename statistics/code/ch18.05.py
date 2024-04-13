import numpy as np

from scipy.stats import chi2

def ll_ratio_test(X):
    X__ = X.sum()
    Xi_ = X.sum(axis=1)
    X_j = X.sum(axis=0)

    T = 2 * sum(
        X[i,j] * np.log(X[i,j] * X__/(Xi_[i] * X_j[j]))
            for i in range(X.shape[0])
            for j in range(X.shape[1])
    )
    df = (X.shape[0] - 1)*(X.shape[1] - 1)
    p_value = 1 - chi2.cdf(T, df)

    return p_value

X = np.array([
    [35, 59, 47, 112],
    [42, 77, 26, 76]
])


# Part 1
p1 = ll_ratio_test(X[:, [0, 2]])
p2 = ll_ratio_test(X[:, [1, 3]])
p_value = max(p1, p2)

print('Test X1 vs X2 | X3, p-value: {:.4f} < 0.025?'.format(p_value))


# Part 2
p1 = ll_ratio_test(X[:, [0, 1]])
p2 = ll_ratio_test(X[:, [2, 3]])
p_value = max(p1, p2)

print('Test X1 vs X3 | X2, p-value: {:.4f} < 0.025?'.format(p_value))


# Part 3
p1 = ll_ratio_test(X[0,:].reshape(2, 2))
p2 = ll_ratio_test(X[1,:].reshape(2, 2))
p_value = max(p1, p2)

print('Test X2 vs X3 | X1, p-value: {:.4f} < 0.025?'.format(p_value))

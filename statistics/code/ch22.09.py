import numpy as np

from sklearn.datasets import load_iris


rng = np.random.RandomState(37)
X, Y = load_iris(return_X_y=True)


def kmean(k, f, X, Y):
    if len(f.shape) > 1:
        return np.array([kmean(k, f_, X, Y) for f_ in f])

    l2 = np.sum((X - f.reshape(1, -1))**2, axis=1)
    l2Yk = sorted(list(zip(l2, Y)))[:k]
    counts = np.bincount([y for _, y in l2Yk])
    return np.argmax(counts)


def cross_validation(k, X=X, Y=Y):
    cv_n = 5
    cv_n_split = X.shape[0] // cv_n
    cv_samples = np.zeros(X.shape[0])
    for i in range(cv_n):
        cv_samples[i*cv_n_split:(i+1)*cv_n_split] = i
    rng.shuffle(cv_samples)

    cv_Y_pred = np.zeros(Y.shape[0])
    for i in range(cv_n):
        cv_X = X[cv_samples != i]
        cv_Y = Y[cv_samples != i]
        cv_X_pred = X[cv_samples == i]
        cv_Y_pred[cv_samples == i] = kmean(k, cv_X_pred, cv_X, cv_Y)

    return sum(cv_Y_pred == Y) / len(Y)


cv_best_k = 0
cv_best_acc = 0
for k in range(1, 20):
    cur_acc = cross_validation(k, X, Y)
    if cur_acc > cv_best_acc:
        cv_best_acc = cur_acc
        cv_best_k = k

print(f'K-means (k={cv_best_k}): {cv_best_acc:.4f}')

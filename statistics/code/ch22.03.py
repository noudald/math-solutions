import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


data = {}

with open('./data/spam.dat', 'r') as f:
    for line in f.readlines():
        features = list(map(float, line.split()))

        if len(data) == 0:
            for i in range(len(features) - 1):
                data[f'x{i}'] = []
            data['y'] = []

        for i, x in enumerate(features[:-1]):
            data[f'x{i}'].append(x)
        data['y'].append(features[-1])

df = pd.DataFrame(data)

features = df.columns[:-1]
X = df[features].values
Y = df['y'].values


# 21.3.(a)
def lda(X, Y, X_pred):
    pi_0 = np.mean(1 - Y)
    pi_1 = np.mean(Y)

    X_0 = X[Y == 0]
    X_1 = X[Y == 1]

    n_0 = len(X_0)
    n_1 = len(X_1)

    mu_0 = np.mean(X_0, axis=0)
    mu_1 = np.mean(X_1, axis=0)

    # Xm_0 = (X_0 - mu_0.reshape(1, -1))
    # Xm_1 = (X_1 - mu_1.reshape(1, -1))
    #
    # S_0 = np.mean([Xm_0[i].reshape(-1, 1) @ Xm_0[i].reshape(-1, 1).T for i in range(len(Xm_0))], axis=0)
    # S_1 = np.mean([Xm_1[i].reshape(-1, 1) @ Xm_1[i].reshape(-1, 1).T for i in range(len(Xm_1))], axis=0)

    S_0 = np.cov(X_0, rowvar=False)
    S_1 = np.cov(X_1, rowvar=False)

    S = (n_0*S_0 + n_1*S_1) / (n_0 + n_1)

    delta_0 = lambda x: x.T @ S**(-1) @ mu_0 - 0.5 * mu_0.T @ S**(-1) @ mu_0 + np.log(pi_0)
    delta_1 = lambda x: x.T @ S**(-1) @ mu_1 - 0.5 * mu_1.T @ S**(-1) @ mu_1 + np.log(pi_1)

    pred = []
    for i in range(len(X_pred)):
        d0 = delta_0(X_pred[i])
        d1 = delta_1(X_pred[i])
        if d1 > d0:
            pred.append(1.0)
        else:
            pred.append(0.0)

    return np.array(pred)


def qda(X, Y, X_pred):
    pi_0 = np.mean(1 - Y)
    pi_1 = np.mean(Y)

    X_0 = X[Y == 0]
    X_1 = X[Y == 1]

    n_0 = len(X_0)
    n_1 = len(X_1)

    mu_0 = np.mean(X_0, axis=0)
    mu_1 = np.mean(X_1, axis=0)

    S_0 = np.cov(X_0, rowvar=False)
    S_1 = np.cov(X_1, rowvar=False)

    S = (n_0*S_0 + n_1*S_1) / (n_0 + n_1)

    delta_0 = lambda x: -0.5 * np.linalg.det(S_0) - 0.5 * (x - mu_0).T @ S**(-1) @ (x - mu_0) + np.log(pi_0)
    delta_1 = lambda x: -0.5 * np.linalg.det(S_1) - 0.5 * (x - mu_1).T @ S**(-1) @ (x - mu_1) + np.log(pi_1)

    pred = []
    for i in range(len(X_pred)):
        d0 = delta_0(X_pred[i])
        d1 = delta_1(X_pred[i])
        if d1 > d0:
            pred.append(1.0)
        else:
            pred.append(0.0)

    return np.array(pred)


def logr(X, Y, X_pred):
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
        if np.abs(ll - ll_new) < 1e-7 or np.isnan(ll_new):
            break
        ll = ll_new

    p_logit_pred = X_pred @ beta.T
    p_pred = 1 / (1 + np.exp(-p_logit_pred))
    y_pred = p_pred.copy()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    return y_pred


def decision_tree_step(X, Y, X_pred, n_step):
    '''A very, very, very slow implementation of decision trees.'''
    n = X.shape[1]  # Number of features
    y_pred = np.zeros(X_pred.shape[0])

    results = []
    for j in range(n):
        Xj = X[:,j]
        for tj in np.unique(Xj):
            p00 = sum(1 - Y[Xj <= tj]) / len(Xj <= tj)
            p01 = sum(Y[Xj <= tj]) / len(Xj <= tj)
            p10 = sum(1 - Y[Xj > tj]) / len(Xj > tj)
            p11 = sum(Y[Xj > tj]) / len(Xj > tj)
            results.append((2 - p00**2 + p01**2 + p10**2 - p11**2, p00, p01, p10, p11, j, tj))

    if len(results) == 0:
        return y_pred

    _, p00, p01, p10, p11, j, tj = sorted(results)[0]

    idx_left = X[:,j] <= tj
    idx_right = X[:,j] > tj

    if (p00 == 1.0 and p01 == 0.0) or n_step == 0:
        y_pred[idx_left] = 0
    else:
        X_left = X[idx_left,:]
        Y_left = Y[idx_left]
        X_pred_left = X_pred[idx_left,:]

        y_pred[idx_left] = decision_tree_step(X_left, Y_left, X_pred_left, n_step - 1)

    if (p11 == 1.0 and p10 == 0.0) or n_step == 0:
        y_pred[idx_right] = 1
    else:
        X_right = X[idx_right,:]
        Y_right = Y[idx_right]
        X_pred_right = X_pred[idx_right,:]

        y_pred[idx_right] = decision_tree_step(X_right, Y_right, X_pred_right, n_step - 1)

    return y_pred


def decision_tree(X, Y, X_pred):
    return decision_tree_step(X, Y, X_pred, 10)


# 22.3.(b)
rng = np.random.RandomState(37)

def cross_validation(classifier, X=X, Y=Y):
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
        cv_Y_pred[cv_samples == i] = classifier(cv_X, cv_Y, cv_X_pred)

    return sum(cv_Y_pred == Y) / len(Y)


# 22.5
def svm(X, Y, X_pred):
    '''I cheat and use third party software.'''
    clf = SVC()
    clf.fit(X, Y)
    return clf.predict(X_pred)


# 22.8
def svm2(X, Y, X_pred, p):
    '''I cheat and use third party software.'''
    clf = SVC(kernel='poly', degree=p, coef0=1.0)
    clf.fit(X, Y)
    return clf.predict(X_pred)


if __name__ == '__main__':
    # Solution 22.3

    lda_pred = lda(X, Y, X)
    print('LDA', sum(lda_pred == Y) / len(Y))

    qda_pred = qda(X, Y, X)
    print('QDA', sum(qda_pred == Y) / len(Y))

    logr_pred = logr(X, Y, X)
    print('Logistic regression', sum(logr_pred == Y) / len(Y))

    dt_pred = decision_tree(X, Y, X)
    print('Decision tree', sum(dt_pred == Y) / len(Y))


    print('Cross validation:')
    print('CV LDA', cross_validation(lda))
    print('CV Logistic regression', cross_validation(logr))


    W = np.abs((X[Y == 0].mean(axis=0) - X[Y == 1].mean(axis=0)) / X.std(axis=0))
    selection = np.array([i for _, i in sorted(list(zip(W, range(len(W)))), reverse=True)][:10])

    X_selection = X[:,selection]

    print('Select 10 features with highest difference between classes:')
    print('CV LDA', cross_validation(lda, X_selection))
    print('CV Logistic regression', cross_validation(logr, X_selection))


    # Solution 22.5

    svm_pred = svm(X, Y, X)
    print('SVM', sum(svm_pred == Y) / len(Y))


    # Solution 22.8

    best_p = 1
    best_p_acc = 0.0
    for p in range(1, 8):
        svm2_cv_acc = cross_validation(lambda X, Y, X_pred: svm2(X, Y, X_pred, p))
        if svm2_cv_acc > best_p_acc:
            best_p = p
            best_p_acc = svm2_cv_acc
        print('HP opt SVM', p, svm2_cv_acc)

    print(f'Best SVM {p}', best_p_acc)


    # 22.11 - Bagging a tree

    # This function is extremely slow. I'll use the sklearn tools instead.
    # dt_pred = decision_tree(X, Y, X)
    # print('Decision tree', sum(dt_pred == Y) / len(Y))


    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X, Y)
    dt_pred = clf.predict(X)
    print('Decision tree (scikit-learn)', sum(dt_pred == Y) / len(Y))

    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=10))
    bag.fit(X, Y)
    bag_pred = bag.predict(X)
    print('Bagging decision tree (scikit-learn)', sum(bag_pred == Y) / len(Y))


    # 22.12 - Boosting a tree

    boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
    boost.fit(X, Y)
    boost_pred = boost.predict(X)
    print('Boost decision tree (scikit-learn)', sum(boost_pred == Y) / len(Y))

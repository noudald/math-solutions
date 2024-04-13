import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(37)
nsim = 100

mean = np.array([3, 8])
cov = np.array([[1, 1], [1, 2]])

samples = rng.multivariate_normal(mean, cov, size=nsim)

def estimates(samples):
    mean_hat = samples.mean(axis=0)
    cov_hat = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            cov_hat[i][j] = 1/(samples.shape[0] - 1) * np.sum(
                (samples[:,i] - mean_hat[i])*(samples[:,j] - mean_hat[j])
            )

    print('Sample mean matrix:')
    print(mean_hat)

    print('Sample covariance matrix:')
    print(cov_hat)

    def correlation(x, y):
        x_mean = x.mean()
        y_mean = y.mean()
        n = x.shape[0]
        x_cor = 1/(n - 1) * np.sum((x - x_mean)**2)
        y_cor = 1/(n - 1) * np.sum((y - y_mean)**2)
        return np.mean((x - x_mean)*(y - y_mean))/np.sqrt(x_cor*y_cor)

    cor_hat = correlation(samples[:,0], samples[:,1])

    print('Sample correlation:')
    print(cor_hat)

    bs_nsim = 10**4
    bs_cor = np.zeros(bs_nsim)
    for i in range(bs_nsim):
        bs_samples = samples[rng.choice(samples.shape[0], size=samples.shape[0], replace=True),:]
        bs_cor[i] = correlation(bs_samples[:,0], bs_samples[:,1])

    print('Bootstrap correlation: [{:.4f}, {:.4f}]'.format(
        np.quantile(bs_cor, 0.025),
        np.quantile(bs_cor, 0.975)
    ))

    theta_hat = 0.5*(np.log(1 + cor_hat) - np.log(1 - cor_hat))
    a, b = theta_hat - 2/np.sqrt(samples.shape[0] - 3), theta_hat + 2/np.sqrt(samples.shape[0] - 3)

    print('Fisher confidence interval for correlation: [{:.4f}, {:.4f}]'.format(
        (np.exp(2*a) - 1)/(np.exp(2*a) + 1), (np.exp(2*b) - 1)/(np.exp(2*b) + 1)
    ))

plt.scatter(samples[:,0], samples[:,1])
plt.show()

estimates(samples)


# Solution 14.6
print('\nSolution 14.6')
samples = rng.multivariate_normal(mean, cov, size=1000)
estimates(samples)

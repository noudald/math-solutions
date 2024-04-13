import numpy as np

from scipy.stats import norm


# Exercise 10.7.a

x = [.225, .262, .217, .240, .230, .229, .235, .217]
y = [.209, .205, .196, .210, .202, .207, .224, .223, .220, .201]

delta_hat = np.mean(x) - np.mean(y)
var_hat = np.var(x) / len(x) + np.var(y) / len(y)
se_hat = np.sqrt(var_hat)

W = delta_hat / se_hat

print(f'P-value: {2 * norm.cdf(-abs(W)):.5f}')
print(f'Confidence interval: ({delta_hat - 2*se_hat:.3f}, {delta_hat + 2*se_hat:.3f})')


# Exercise 10.7.b

num_simulations = 10**5
rng = np.random.RandomState(37)

data = np.concatenate([x, y])
t_obs = abs(delta_hat)
t_perms = np.empty([num_simulations])
for i in range(num_simulations):
    rng.shuffle(data)
    x_ = data[0:len(x)]
    y_ = data[len(x):]
    t_perms[i] = abs(np.mean(x_) - np.mean(y_))

print(f'P-value: {np.sum(t_perms > t_obs) / num_simulations:.5f}')



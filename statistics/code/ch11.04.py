import numpy as np

from scipy.stats import norm

n1 = 50
n2 = 50
x1 = 30
x2 = 40

# 11.4.(a)
p1_hat = x1/n1
p2_hat = x2/n2
tau_hat = p1_hat - p2_hat

se_hat = np.sqrt(p1_hat*(1 - p1_hat)/n1 + p2_hat*(1 - p2_hat)/n2)

print('Tau 90% confidence interval (Delta method): ({:.2f}, {:.2f})'.format(
    tau_hat - norm.ppf(0.95)*se_hat,
    tau_hat + norm.ppf(0.95)*se_hat
))


# 11.4.(b)
pbs_rng = np.random.RandomState(37)
pbs_nsim = 10**4
pbs_x1 = pbs_rng.binomial(n1, p1_hat, size=pbs_nsim)
pbs_x2 = pbs_rng.binomial(n2, p2_hat, size=pbs_nsim)
pbs_tau = pbs_x1/n1 - pbs_x2/n2

print('Tau 90% confidence interval (Parametric bootstrap): ({:.2f}, {:.2f})'.format(
    np.quantile(pbs_tau, 0.05), np.quantile(pbs_tau, 0.95)
))


# 11.4.(c)
post_rng = np.random.RandomState(37)
post_nsim = 10**4
post_p1 = post_rng.beta(x1 + 1, n1 - x1 + 1, size=post_nsim)
post_p2 = post_rng.beta(x2 + 1, n2 - x2 + 1, size=post_nsim)
post_tau = post_p1 - post_p2

print('Tau 90% confidence interval (Bayesian posterior): ({:.2f}, {:.2f})'.format(
    np.quantile(post_tau, 0.05), np.quantile(post_tau, 0.95)
))


# 11.4.(d)
def psi(p1, p2):
    return np.log((p1/(p1*(1 - p1))) / (p2/(p2*(1 - p2))))

psi_hat = psi(p1_hat, p2_hat)
se_hat = np.sqrt(1/(n1*p1_hat*(1 - p1_hat)) + 1/(n2*p2_hat*(1 - p2_hat)))

print('Psi 90% confidence interval (Delta method): ({:.2f}, {:.2f})'.format(
    psi_hat - norm.ppf(0.95)*se_hat,
    psi_hat + norm.ppf(0.95)*se_hat
))

pbs_psi = psi(pbs_x1/n1, pbs_x2/n2)

print('Psi 90% confidence interval (Parametric bootstrap): ({:.2f}, {:.2f})'.format(
    np.quantile(pbs_psi, 0.05), np.quantile(pbs_psi, 0.95)
))

post_psi = psi(post_p1, post_p2)

print('Psi 90% confidence interval (Bayesian posterior): ({:.2f}, {:.2f})'.format(
    np.quantile(post_psi, 0.05), np.quantile(post_psi, 0.95)
))

import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(37)

init = 0
a = 0.1
b = 0.3
n_sim = 10**5
results = [init]
state = init

for _ in range(n_sim):
    if state == 0:
        if rng.random() < a:
            state = 1
    elif state == 1:
        if rng.random() < b:
            state = 0
    results.append(state)

results = np.array(results)

p1_hat = np.cumsum(1 - results[1:]) / np.arange(1, n_sim + 1)
p2_hat = 1 - p1_hat

plt.plot(np.arange(1, n_sim + 1), p1_hat)
plt.plot(np.arange(1, n_sim + 1), p2_hat)
plt.hlines(a / (a + b), 1, n_sim, color='orange', linestyle='--')
plt.hlines(b / (a + b), 1, n_sim, color='blue', linestyle='--')
plt.show()

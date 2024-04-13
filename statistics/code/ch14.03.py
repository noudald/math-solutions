# I make it myself very easy.

import numpy as np

rng = np.random.RandomState(37)
nsim = 10

print(rng.multinomial(100, [1/2, 1/4, 1/4], size=nsim))

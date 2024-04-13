import numpy as np

beta = 0.5
rng = np.random.RandomState(37)

u = rng.uniform(0, 1)
x = -beta * np.log(1 - u)

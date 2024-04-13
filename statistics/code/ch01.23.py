import numpy as np

n = 10000

A = [2, 4, 6]
B = [1, 2, 3, 4]
AB = [2, 4]

rng = np.random.RandomState(37)
dice = rng.randint(1, 7, size=(n,))

pA_hat = sum(np.isin(dice, A)) / n
pB_hat = sum(np.isin(dice, B)) / n
pAB_hat = sum(np.isin(dice, AB)) / n

print('Independent events simulation:')
print(f'P(A)     ~= {pA_hat:.3f}')
print(f'P(B)     ~= {pB_hat:.3f}')
print(f'P(AB)    ~= {pAB_hat:.3f}')
print(f'P(A)P(B) ~= {pA_hat*pB_hat:.3f}')


A = [1]
B = [1]
AB = [1]

pA_hat = sum(np.isin(dice, A)) / n
pB_hat = sum(np.isin(dice, B)) / n
pAB_hat = sum(np.isin(dice, AB)) / n

print('Dependent events simulation:')
print(f'P(A)     ~= {pA_hat:.3f}')
print(f'P(B)     ~= {pB_hat:.3f}')
print(f'P(AB)    ~= {pAB_hat:.3f}')
print(f'P(A)P(B) ~= {pA_hat*pB_hat:.3f}')

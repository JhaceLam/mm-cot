import numpy as np

a = np.array([0, 1, 2, 3, 4])
id_0 = id(a)
b = np.array([5, 6, 7, 8, 9])

a = np.concatenate((a, b))
print(a)
id_1 = id(a)

b[2] = 10
print(b)
print(a)
import numpy as np
import matplotlib.pyplot as plt

a = np.array([2, 1, 4, 1, 3, 10, 30, 15, 20, 5])
b = a.max() - a.min()
n = 3
index = []
for i in range(n):
    temp = np.where(a > (a.min() + i * b / n))[0]
    index.append(temp.shape[0])

print index


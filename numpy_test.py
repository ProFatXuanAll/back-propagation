import numpy as np

a = np.matrix([[1,2,3],[4,5,6]])

def f(x):
    return 1/(1 + np.exp(-x))

for _ in a:
    print(_)
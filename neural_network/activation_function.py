import math

def sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))

sigmoid.derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))

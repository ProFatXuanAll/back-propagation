import numpy as np

def sigmoid(x, *args, **kwargs):
    return 1 / (1 + np.exp(-x))

def __d_sigmoid(x, *args, **kwargs):
    return sigmoid(x) * (1 - sigmoid(x))
sigmoid.derivative = __d_sigmoid

def relu(x, *args, **kwargs):
    if x > 0:
        return x
    return 0

def __d_relu(x, *args, **kwargs):
    if x > 0:
        return 1
    return 0

relu.derivative = __d_relu

def softmax(x, row, *args, **kwargs):
    base = np.sum(np.exp(row))
    return np.exp(x) / base

def __d_softmax(x, row, *args, **kwargs):
    base = np.sum(np.exp(row))
    return (np.exp(x)*base - np.exp(x)*np.exp(x)) / (base * base)

softmax.derivative = __d_softmax
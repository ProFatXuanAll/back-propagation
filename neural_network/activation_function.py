import math

class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        return 1 / (1 + math.exp(-1 * x))

    @staticmethod
    def derivative(x):
        y = Sigmoid.function(x)
        return y * (1 - y)
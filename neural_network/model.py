import numpy as np

class Model:
    def __init__(self, batch_size=1, epoch=1):
        self.__layer = []
        self.__batch_size = batch_size
        self.__epoch = epoch

    def add(self, layer):
        self.__layer.append(layer)

    def predict(self, x):
        for layer in self.__layer:
            x = layer.forward_pass(x)
        return x

    def update(self, E):
        for i in range(len(self.__layer)):
            E = self.__layer[i].back_propagate(E)

    def train(self, x_list, y_list):
        for x, y in x_list, y_list:
            E = self.predict(x) - y
            self.update(E)

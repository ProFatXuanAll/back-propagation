import math
import random
import numpy as np

class Model:
    def __init__(self, error_f, batch_size=1, epoch=1):
        self.__layer = []
        self.__error_f = error_f
        self.__batch_size = batch_size
        self.__epoch = epoch

    def add(self, layer):
        self.__layer.append(layer)

    def predict(self, x):
        for layer in self.__layer:
            x = layer.forward_pass(x)
        return x

    def error(self, x, y):
        return self.__error_f(self.predict(x), y)

    def error_derivative(self, x, y):
        return self.__error_f.derivative(self.predict(x), y)

    def get_batch(self, data, i):
        try:
            batch = data[i*self.__batch_size:(i+1)*self.__batch_size]
        except:
            batch = data[i*self.__batch_size:]
        return batch

    def train(self, x, y):
        for i in range(self.__epoch):
            print('epoch: {}/{}'.format(i+1, self.__epoch))
            batch = math.ceil(x.shape[0]/self.__batch_size)
            for j in range(batch):
                progress_percentage = (j+1)/batch*100
                progress = math.floor(progress_percentage/10)
                progress_bar = ''.join(['=' for _ in range(progress-1)]) + '>' + ''.join(' ' for _ in range(10-progress))
                print('batch: [{}] {:2.2f}%'.format(progress_bar, progress_percentage), end='\r')

                # SGD selection
                _x = self.get_batch(x, j)
                _y = self.get_batch(y, j)
                sgd_index = math.floor(random.uniform(0, _x.shape[0]))
                _x = _x[sgd_index]
                _y = _y[sgd_index]
                dE_over_dy = self.error_derivative(_x, _y)

                # update section
                for which_layer in range(len(self.__layer)):
                    _x = self.__layer[which_layer].remember(_x)
                for which_layer in range(len(self.__layer)-1, -1, -1):
                    dE_over_dy = self.__layer[which_layer].back_propagate(dE_over_dy)
            # show error
            print('\nerror: {}'.format(np.sum(self.error(x, y), axis=0)))


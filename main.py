import numpy as np
from neural_network.model import Model
from neural_network.layer import FullyConnected
from neural_network.activation_function import Sigmoid

w1 = np.matrix([[1,4,7],[2,3,8],[5,6,9]])
b1 = np.matrix([[-12, -13, -20]])
f1 = np.array([Sigmoid, Sigmoid, Sigmoid])

w2 = np.matrix([[1,4,7],[2,3,8],[5,6,9]])
b2 = np.matrix([[-6, -6.5, -10]])
f2 = np.array([Sigmoid, Sigmoid, Sigmoid])

eta = 1
alpha = 1
input = np.matrix([[1, 1, 1], [2, 2, 2]])

m1 = Model()
m1.add(FullyConnected(x_dim=3, y_dim=3, f=f1, w=w1, b=b1, eta=eta, alpha=alpha))
# m1.add(FullyConnected(x_dim=3, y_dim=3, f=f2, w=w2, b=b2, eta=eta, alpha=alpha))

print(m1.predict(input))
import numpy as np
from neural_network.model import Model
from neural_network.layer import FullyConnected
from neural_network.activation_function import sigmoid
from neural_network.error_function import sq_error

x = np.matrix([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
])
y = np.matrix([
    [1,1],
    [0,1],
    [1,0],
    [0,0],
    [1,0],
    [0,0],
    [1,1],
    [0,1],
    [0,1],
    [1,1],
    [0,0],
    [1,0],
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])
w = np.matrix([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])
b = np.matrix([0, 0, 0])
f = np.array([sigmoid, sigmoid, sigmoid])
eta = 1
alpha = 1

batch_size = 1
epoch = 3
m = Model(error_f=sq_error, batch_size=batch_size, epoch=epoch)
m.add(FullyConnected(x_dim=4, y_dim=3, f=f, w=w, b=b, eta=eta, alpha=alpha))

w = np.matrix([
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1]
])
b = np.matrix([0, 0, 0, 0, 0])
f = np.array([sigmoid, sigmoid, sigmoid, sigmoid, sigmoid])

m.add(FullyConnected(x_dim=3, y_dim=5, f=f, w=w, b=b, eta=eta, alpha=alpha))

w = np.matrix([
    [1,1,1,1,1],
    [1,1,1,1,1]
])
b = np.matrix([0, 0])
f = np.array([sigmoid, sigmoid])

m.add(FullyConnected(x_dim=5, y_dim=2, f=f, w=w, b=b, eta=eta, alpha=alpha))

m.train(x,y)
# print('predict')
# print(m.predict(x))

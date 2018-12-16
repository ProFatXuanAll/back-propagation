import numpy as np
from neural_network.model import Model
from neural_network.layer import FullyConnected
from neural_network.activation_function import sigmoid
from neural_network.error_function import sq_error

x = np.matrix([
    [0,0,0,0,0,0],
    [0,0,0,0,0,1],
    [0,0,0,0,1,0],
    [0,0,0,0,1,1],
    [0,0,0,1,0,0],
    [0,0,0,1,0,1],
    [0,0,0,1,1,0],
    [0,0,0,1,1,1],
    [0,0,1,0,0,0],
    [0,0,1,0,0,1],
    [0,0,1,0,1,0],
    [0,0,1,0,1,1],
    [0,0,1,1,0,0],
    [0,0,1,1,0,1],
    [0,0,1,1,1,0],
    [0,0,1,1,1,1],
    [0,1,0,0,0,0],
    [0,1,0,0,0,1],
    [0,1,0,0,1,0],
    [0,1,0,0,1,1],
    [0,1,0,1,0,0],
    [0,1,0,1,0,1],
    [0,1,0,1,1,0],
    [0,1,0,1,1,1],
    [0,1,1,0,0,0],
    [0,1,1,0,0,1],
    [0,1,1,0,1,0],
    [0,1,1,0,1,1],
    [0,1,1,1,0,0],
    [0,1,1,1,0,1],
    [0,1,1,1,1,0],
    [0,1,1,1,1,1],
    [1,0,0,0,0,0],
    [1,0,0,0,0,1],
    [1,0,0,0,1,0],
    [1,0,0,0,1,1],
    [1,0,0,1,0,0],
    [1,0,0,1,0,1],
    [1,0,0,1,1,0],
    [1,0,0,1,1,1],
    [1,0,1,0,0,0],
    [1,0,1,0,0,1],
    [1,0,1,0,1,0],
    [1,0,1,0,1,1],
    [1,0,1,1,0,0],
    [1,0,1,1,0,1],
    [1,0,1,1,1,0],
    [1,0,1,1,1,1],
    [1,1,0,0,0,0],
    [1,1,0,0,0,1],
    [1,1,0,0,1,0],
    [1,1,0,0,1,1],
    [1,1,0,1,0,0],
    [1,1,0,1,0,1],
    [1,1,0,1,1,0],
    [1,1,0,1,1,1],
    [1,1,1,0,0,0],
    [1,1,1,0,0,1],
    [1,1,1,0,1,0],
    [1,1,1,0,1,1],
    [1,1,1,1,0,0],
    [1,1,1,1,0,1],
    [1,1,1,1,1,0],
    [1,1,1,1,1,1],
])
y = np.matrix([
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
    [1],
])
batch_size = 1
epoch = 1425

w1 = np.matrix([
    [-0.3,-0.2,-0.1,0.1,0.2,0.3],
    [0.3,0.2,0.1,-0.1,-0.2,-0.3]
])
b1 = np.matrix([0, 0])
f1 = np.array([sigmoid, sigmoid])
w2 = np.matrix([
    [1,1]
])
b2 = np.matrix([0])
f2 = np.array([sigmoid])

eta = 0.1
alpha = 0.9

m = Model(error_f=sq_error, batch_size=batch_size, epoch=epoch)
m.add(FullyConnected(x_dim=6, y_dim=2, f=f1, w=w1, b=b1, eta=eta, alpha=alpha))
m.add(FullyConnected(x_dim=2, y_dim=1, f=f2, w=w2, b=b2, eta=eta, alpha=alpha))
m.train(x,y)

print(m.predict(x))

import numpy as np
from neural_network.model import Model
from neural_network.layer import FullyConnected
from neural_network.activation_function import sigmoid

# test forward pass
# wx = np.matrix([
#   [-40, 0, 70],
#   [-44, 0, 76],
#   [-68, 0, 92]
# ])
#
# wx_b = np.matrix([
#   [   0   40  110],
#   [ -44    0   76],
#   [-160  -92    0]
# ])
#
# f(wx_b) = np.matrix([
#   [5.00000000e-01 1.00000000e+00 1.00000000e+00]
#   [7.78113224e-20 5.00000000e-01 1.00000000e+00]
#   [3.25748853e-70 1.10893902e-40 5.00000000e-01]
# ])
#
#
#

x = np.matrix([
    [-4,-3,-2,-1],
    [0,0,0,0],
    [1,2,3,4]
])
w = np.matrix([
    [1,4,7,10],
    [2,3,8,11],
    [5,6,9,12]
])
b = np.matrix([40, 0, -92])
f = np.array([sigmoid, sigmoid, sigmoid])
eta = 1
alpha = 1

m = Model()
m.add(FullyConnected(x_dim=4, y_dim=3, f=f, w=w, b=b, eta=eta, alpha=alpha))

print(m.predict(x))

w = np.matrix([
    [-6,-5,-4],
    [-3,-2,-1],
    [0,0,0],
    [1,2,3],
    [4,5,6]
])
b = np.matrix([12,4.5,0,-1.5,-3])
f = np.array([sigmoid, sigmoid, sigmoid, sigmoid, sigmoid])
eta = 1
alpha = 1
m.add(FullyConnected(x_dim=3, y_dim=5, f=f, w=w, b=b, eta=eta, alpha=alpha))

print(m.predict(x))
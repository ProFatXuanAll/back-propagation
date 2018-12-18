import numpy as np
from neural_network.model import Model
from neural_network.layer import FullyConnected
from neural_network.activation_function import sigmoid, relu
from neural_network.error_function import sq_error

from keras.models import Sequential
from keras.layers import Dense

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
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
])
batch_size = 1
epoch = 1425

w1 = np.matrix([
    [-0.3,0.2,-0.1,0.6,0.7,-0.4],
    [0.3,-0.2,0.3,-0.3,-0.8,-0.5]
])
b1 = np.matrix([0, 0])
f1 = np.array([sigmoid, sigmoid])
w2 = np.matrix([
    [1,1]
])
b2 = np.matrix([0])
f2 = np.array([sigmoid])

eta = 0.01
alpha = 1

m = Model(error_f=sq_error, batch_size=batch_size, epoch=epoch)
m.add(FullyConnected(x_dim=6, y_dim=2, f=f1, w=w1, b=b1, eta=eta, alpha=alpha))
m.add(FullyConnected(x_dim=2, y_dim=1, f=f2, w=w2, b=b2, eta=eta, alpha=alpha))
m.train(x,y)

for layer in m._Model__layer:
    print('weight')
    print(layer._FullyConnected__w)
    print('bias')
    print(layer._FullyConnected__b)

print('predict:')
print(m.predict(x))

### Compare to other package

# model = Sequential()
# model.add(Dense(units=2, activation='sigmoid', input_dim=6))
# model.add(Dense(units=1, activation='sigmoid'))
# model.compile(loss='mean_squared_error', optimizer='sgd')
# model.fit(x,y,epochs=epoch,batch_size=batch_size)
# print(model.evaluate(x,y))
# print('actual_y | predict_class | predict_y')
# print('------------------------------------')
# for actual_y, predict_y in zip(y, model.predict(x)):
#     print('{:1.6f} | {:1.6f}      | {:1.6f}'.format(actual_y[0,0], round(predict_y[0]), predict_y[0]))


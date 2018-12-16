'''Layer of neural network.'''
import numpy as np

class FullyConnected:
    '''Fully Connected Layer.'''
    def __init__(self,
                 x_dim=0,
                 y_dim=0,
                 f=None,
                 w=None,
                 b=None,
                 eta=None,
                 alpha=None):
        '''Initialize fully connected layer.

        Args:
            x_dim (int): Input dimension.
            y_dim (int): Output dimension.
            f (list of function): List of activation functions.
            w (matrix of float): Matrix of initial weights.
            b (list of float): List of initial bias.
            eta (float): List of learning rate.
            alpha (float): List of normalization term.
        '''
        self.__x_dim = x_dim
        self.__y_dim = y_dim
        self.__x = None
        self.__y = None
        self.__f = f
        self.__w = np.matrix(w)
        self.__b = np.matrix(b)
        self.__wxb = None
        self.__eta = eta
        self.__alpha = alpha

    def forward_pass(self, x):
        """Forward pass.

        Args:
            x (list of float): Input vector.

        Returns:
            list of float: Output.
        """
        # Calculate wx+b
        self.__x = np.matrix(x)
        print('x:')
        print(self.__x)
        self.__wxb = self.__w * np.transpose(self.__x) + np.transpose(self.__b)
        print('wx:')
        print(self.__w * np.transpose(self.__x))
        print('b:')
        print(np.transpose(self.__b))
        print('wx+b:')
        print(self.__wxb)

        self.__y = []
        for f, row in zip(self.__f, self.__wxb.tolist()):
            self.__y.append([f(col) for col in row])

        self.__y = np.matrix(self.__y)
        return self.__y

    def back_propagate(self, last__dE__over__dx):
        """Back propagation algorithm.

        Args:
            input (list of float): Input vector.

        Returns:
            tuple of float: Output.
        """
        pass
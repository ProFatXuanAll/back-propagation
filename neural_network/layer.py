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
        self.__f = f
        self.__w = np.matrix(w)
        self.__b = np.matrix(b)
        self.__eta = eta
        self.__alpha = alpha

        self.__x = None
        self.__wxb = None
        self.__y = None

    def wxb(self, x):
        return self.__w * np.transpose(x) + np.transpose(self.__b)

    def forward_pass(self, x):
        """Forward pass.

        Args:
            x (list of float): Input vector.

        Returns:
            list of float: Output.
        """
        # Calculate wx+b
        x = np.matrix(x)
        wxb = self.wxb(x)

        y = []
        for f, row in zip(self.__f, wxb.tolist()):
            y.append([f(col) for col in row])

        y = np.matrix(y)
        return np.transpose(y)

    def remember(self, x):
        self.__x = np.matrix(x)
        self.__wxb = self.wxb(x)
        self.__y = self.forward_pass(x)
        return self.__y

    def back_propagate(self, dE_over_dy):
        """Back propagation algorithm.

        Args:
            input (list of float): Input vector.

        Returns:
            tuple of float: Output.
        """
        # Gradient over weight
        dwxb_over_dw = self.__x
        for _ in range(self.__y_dim-1):
            dwxb_over_dw = np.concatenate((dwxb_over_dw, self.__x))

        dy_over_dwxb = np.array([f.derivative(wxb) for f, wxb in zip(self.__f, self.__wxb)])
        dy_over_dw = dy_over_dwxb[0] * dwxb_over_dw[0]
        for i in range(1, self.__y_dim):
            dy_over_dw = np.concatenate([dy_over_dw, dy_over_dwxb[i] * dwxb_over_dw[i]])

        dE_over_dw = dE_over_dy[0, 0] * dy_over_dw[0]
        for i in range(1, self.__y_dim):
            dE_over_dw = np.concatenate([dE_over_dw, dE_over_dy[0, i] * dy_over_dw[i]])

        # Graident over input
        dwxb_over_dx = self.__w

        dy_over_dx = dy_over_dwxb[0] * dwxb_over_dx[0]
        for i in range(1, self.__y_dim):
            dy_over_dx = np.concatenate([dy_over_dx, dy_over_dwxb[i] * dwxb_over_dx[i]])

        dE_over_dx = dE_over_dy[0, 0] * dy_over_dx[0]
        for i in range(1, self.__y_dim):
            dE_over_dx = np.concatenate([dE_over_dx, dE_over_dy[0, i] * dy_over_dx[i]])

        # update weight
        self.__w = self.__alpha * self.__w - self.__eta * dE_over_dw

        return np.matrix(np.sum(dE_over_dx, axis=0))
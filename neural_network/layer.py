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
            eta (float): Learning Rate.
            alpha (float): Normalization Term.
        '''
        self.__x_dim = x_dim
        self.__x = np.ones((1, self.__x_dim))
        self.__y_dim = y_dim
        self.__y = np.ones((1, self.__y_dim))
        self.__f = f
        self.__w = w
        self.__b = b
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
        wxb = self.__w * x.T + self.__b.T

        # Remember current input which will be used in back propagation.
        self.__x = x
        # Remember current output which will be used in back propagation.
        self.__y = []

        # Calculate f(wx+b)
        for i in range(self.__y_dim):
            self.__y.append(self.__f[i].function(wxb[i]))

        self.__y = np.array(self.__y)
        return self.__y

    def back_propagate(self, last__dE__over__dx):
        """Back propagation algorithm.

        Args:
            input (list of float): Input vector.

        Returns:
            tuple of float: Output.
        """

        # Calculate gradient of weights (dE over dw).
        dE__over__dw = []
        for i in range(self.__y_dim):
            dE__over__dy_i = last__dE__over__dx[i]
            dy__over__dwxb_i = self.__f[i].derivative(self.__y[i])

            dE__over__dw_i = []
            for j in range(self.__x_dim):
                dwxb_i__over__dw_i_j = self.__x[j]
                dE__over__dw_i_j = dE__over__dy_i * dy__over__dwxb_i * dwxb_i__over__dw_i_j
                dE__over__dw_i.append(dE__over__dw_i_j)
            dE__over__dw.append(dE__over__dw_i)

        dE__over__dw = np.matrix(dE__over__dw)

        # Back propagate gradient of x (dE over dx[i]).
        dE__over__dx = []
        dE__over__dy_0 = last__dE__over__dx[0]
        dy_0__over__dwxb = self.__f[0].derative(self.__y[0])
        for i in range(self.__x_dim):
            dwxb__over__dx_i = self.__w[0,i]
            dE__over__dx.append(dE__over__dy_0 * dy_0__over__dwxb * dwxb__over__dx_i)

        # Update weights.
        self.__w = self.__eta * dE__over__dw + self.__alpha * self.__w

        return np.array(dE__over__dx)

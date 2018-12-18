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

    def weighted_sum(self, x):
        """Weighted sum $w * x + b$.

        Args:
            x (matrix of float): Input matrix.

        Returns:
            matrix of float: Weighted sum.
        """
        # Copy `w` and convert into matrix, dimension (m, d).
        w = np.matrix(self.__w)
        # Copy `x` and convert into matrix, dimension (n, d).
        x = np.matrix(x)
        # Transpose `x`, dimension (d, n).
        x = np.transpose(x)
        # Copy `b` and convert into matrix, dimension (1, m).
        b = np.matrix(self.__b)
        # Transpose `b`, dimension (m, 1).
        b = np.transpose(b)
        # Duplicate `b` n times and combine into one matrix, dimension (m, n).
        b = np.concatenate([b for _ in range(x.shape[1])], axis=1)
        # Weighted sum, dimension (m, d) * (d, n) + (m, n) = (m, n).
        return w * x + b

    def forward_pass(self, x):
        """Forward pass.

        Args:
            x (list of float): Input vector.

        Returns:
            list of float: Output.
        """
        # Copy `w * x + b` and convert into matrix, dimension (m, n).
        wx_b = np.matrix(self.weighted_sum(x))
        # Copy `f` and convert into array, dimension(1, m)
        f = np.array(self.__f)
        # Activated matrix `y` which element are activated by `f` with input `wx_b`, dimension (m, n).
        y = np.matrix(np.zeros(wx_b.shape))

        # Perform activation.
        for row in range(wx_b.shape[0]):
            for col in range(wx_b.shape[1]):
                y[row, col] = f[row](wx_b[row, col], wx_b[row, :])
        # Transpose `y`, dimension (n, m).
        return np.transpose(y)

    def remember(self, x):
        self.__x = np.matrix(x)
        self.__wxb = self.weighted_sum(x)
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

        # Gradient over bias
        # dE_over_db = np.multiply(dE_over_dy,dy_over_dwxb)

        # Gradient over input
        dwxb_over_dx = self.__w

        dy_over_dx = dy_over_dwxb[0] * dwxb_over_dx[0]
        for i in range(1, self.__y_dim):
            dy_over_dx = np.concatenate([dy_over_dx, dy_over_dwxb[i] * dwxb_over_dx[i]])

        dE_over_dx = dE_over_dy[0, 0] * dy_over_dx[0]
        for i in range(1, self.__y_dim):
            dE_over_dx = np.concatenate([dE_over_dx, dE_over_dy[0, i] * dy_over_dx[i]])

        # update weight
        self.__w = self.__alpha * self.__w - self.__eta * dE_over_dw

        # update bias
        # self.__b = self.__alpha * self.__b - self.__eta * dE_over_db

        return np.matrix(np.sum(dE_over_dx, axis=0))
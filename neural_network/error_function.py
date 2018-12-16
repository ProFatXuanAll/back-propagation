import numpy as np
import math

# for vector
def sq_error(predict_y, actual_y):
    diff = predict_y - actual_y
    error = np.concatenate([np.multiply(d, d) for d in diff])
    return 0.5 * error

# for scalar
sq_error.derivative = lambda predict_y, actual_y: predict_y - actual_y

import numpy as np


def get_rand_linear_function(low_range, high_range):
    '''
    Returns:
        random m, b for the equation y = mx + b
    '''
    rand_points = np.random.uniform(
        low=low_range,
        high=high_range,
        size=(2, 2),
    )

    a = rand_points[0]
    b = rand_points[1]

    # m = slope = (y1-y2)/(x1-x2)
    m = (a[1] - b[1]) / (a[0] - b[0])

    # b = y-intercept = (x1*y1 - x2*y1)/(x1-x2)
    # b =  y - mx
    b = a[1] - m*a[0]

    return m, b


def create_labeled_points(func_m, func_bias, num_points, low_range, high_range, d):
    # creates a random linear target function
    X = np.random.uniform(
        low=low_range,
        high=high_range,
        size=(num_points, d),
    )

    y = np.zeros((num_points))

    for i in range(num_points):
        if X[i, 1] > func_m*X[i, 0] + func_bias:
            y[i] = 1
        else:
            y[i] = -1

    return X, y


def evaluate(y_pred, y_true):
    num_points = len(y_true)
    total_disagree = len(np.where(y_true != y_pred)[0])
    avg_disagree = total_disagree / num_points
    return avg_disagree


class LinearPredictor:
    def __init__(self, func_m, func_b):
        self.func_m = func_m
        self.func_b = func_b

    def predict(self, X):
        y = np.zeros((len(X)))

        for i in range(len(X)):
            if X[i, 1] > self.func_m*X[i, 0] + self.func_b:
                y[i] = 1
            else:
                y[i] = -1

        return y

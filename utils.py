import numpy as np
import matplotlib.pyplot as plt

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
    print('total_disagree', total_disagree)
    print('num_points', num_points)
    avg_disagree = total_disagree / num_points
    return avg_disagree


class LinearPredictor:
    def __init__(self, func_m, func_b):
        self.func_m = func_m
        self.func_b = func_b

    def predict(self, X):
        y = np.zeros((len(X)))

        for i in range(len(X)):
            if X[i, 1] >= self.func_m*X[i, 0] + self.func_b:
                y[i] = 1
            else:
                y[i] = -1

        return y


def linear_regression(X, y, should_plot=False):
    X0 = np.ones((len(X), 1))

    # adding bias to X
    X = np.hstack((X0, X))

    X_hat = np.linalg.inv(X.T.dot(X)).dot(X.T)
    w = X_hat.dot(y)

    decision_boundary_m = -w[1] / w[2]
    decision_boundary_b = -w[0] / w[2]

    X_1 = X[:, 1]
    X_2 = decision_boundary_m * X_1 + decision_boundary_b

    if should_plot:
        plt.plot(X_1, X_2, 'r')

    return decision_boundary_m, decision_boundary_b


def plot_points(X, y, first_color, second_color):
    first_label_dx = np.where(y == -1)
    plt.scatter(
        X[first_label_dx, 0],
        X[first_label_dx, 1],
        color=first_color,
    )
    second_label_dx = np.where(y == 1)

    plt.scatter(
        X[second_label_dx, 0],
        X[second_label_dx, 1],
        color=second_color,
    )

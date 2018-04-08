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


def create_points(num_points, low_range, high_range, d):
    # creates a random linear target function
    target_func_m, target_func_b = get_rand_linear_function(
        low_range=low_range,
        high_range=high_range,
        )

    X = np.random.uniform(
        low=low_range,
        high=high_range,
        size=(num_points, d),
    )

    y = np.zeros((num_points))

    for i in range(num_points):
        if X[i, 1] > target_func_m*X[i, 0] + target_func_b:
            y[i] = 1
        else:
            y[i] = -1

    return X, y, target_func_m, target_func_b


def evaluate(num_points, pred_func_m, pred_func_b, target_func_m, target_func_b, low_range, high_range, d):
    # creates a random linear target function
    # target_func_m, target_func_b = get_rand_linear_function()

    X = np.random.uniform(
        low=low_range,
        high=high_range,
        size=(num_points, d),
    )

    target_y = np.zeros((num_points))

    for i in range(num_points):
        if X[i, 1] > target_func_m*X[i, 0] + target_func_b:
            target_y[i] = 1
        else:
            target_y[i] = -1

    pred_y = np.zeros((num_points))

    for i in range(num_points):
        if X[i, 1] > pred_func_m*X[i, 0] + pred_func_b:
            pred_y[i] = 1
        else:
            pred_y[i] = -1

    total_disagree = len(np.where(target_y != pred_y)[0])
    avg_disagree = total_disagree / num_points
    print('avg_disagree', avg_disagree)

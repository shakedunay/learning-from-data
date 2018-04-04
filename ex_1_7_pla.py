import numpy as np
import matplotlib.pyplot as plt

low_range = -1
high_range = 1

d = 2

def get_rand_linear_function():
    '''
    Returns:
        random m, b for the equation y = mx + b
    '''
    rand_points = np.random.uniform(
        low=low_range,
        high=high_range,
        size=(2,2),
    )

    a = rand_points[0]
    b = rand_points[1]

    # m = slope = (y1-y2)/(x1-x2)
    m = (a[1] - b[1]) / (a[0] - b[0])

    # b = y-intercept = (x1*y1 - x2*y1)/(x1-x2)
    # b =  y - mx
    b = a[1] - m*a[0]

    return m, b 


def create_points(num_points):
    # creates a random linear target function
    target_func_m, target_func_b = get_rand_linear_function()
    
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
        

def perceptronCalc(x, w):
    return x[0]*w[0] + x[1]*w[1] + x[2]*w[2]

def sign(x, w):
    if perceptronCalc(x, w) >= 0:
        return 1
    else:
        return -1

def change_weights(X, y, w):
    weights_changed = False
    for i in range(len(X)):
        should_change = sign(X[i], w) != y[i]
        if should_change:
            weights_changed = True
            w[0] += X[i, 0]*y[i]
            w[1] += X[i, 1]*y[i]
            w[2] += X[i, 2]*y[i]
    return weights_changed

def pla(X, y, should_plot=False):
    '''
    Returns:
        pred_m, pred_b, num_iterations
        pred_m, pred_b - the slop the y-intersection that solves that linear seperates the given points
        num_iterations - num iteration it took the pla to run
    '''
    num_iterations = 0 
    # pred_m, pred_b = get_rand_linear_function()
    should_run = True

    X0 = np.ones((len(X), 1))

    # adding bias to X
    X_new = np.hstack((X0, X))
    w = np.ones((3,1))
    while should_run:
        weights_changed = change_weights(X_new, y, w)
        num_iterations += 1
        should_run = weights_changed
    
    # As your decision function is simply sign(w0 + w1*x1 +w2*x2) then the decision boundary equation is a line with canonical form: w0 + w1*x1 + w2*x2 = 0
    decision_boundary_m = -w[1] / w[2]
    decision_boundary_b = -w[0] / w[2]

    X_1 = X_new[:, 1]
    X_2 = decision_boundary_m * X_1 + decision_boundary_b    
    
    if should_plot:
        plt.plot(X_1, X_2 , 'r')

    return num_iterations, decision_boundary_m, decision_boundary_b

def main():
    num_experiments = 1
    total_iterations = 0
    
    should_plot = True

    for _ in range(num_experiments):
        X, y, target_func_m, target_func_b = create_points(num_points=100)
        

        if should_plot:
            first_label_dx = np.where(y == -1)
            plt.scatter(
                X[first_label_dx, 0],
                X[first_label_dx, 1],
            )
            second_label_dx = np.where(y == 1)

            plt.scatter(
                X[second_label_dx, 0],
                X[second_label_dx, 1],
            )

        num_iterations, pred_m, pred_b = pla(X, y, should_plot)

        total_iterations += num_iterations

        # print(pred_m, target_func_m)
        # print(pred_b, target_func_b)

    avg_iterations = total_iterations / num_experiments
    print(avg_iterations)
    plt.show()

if __name__ == '__main__':
    main()

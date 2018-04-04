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
        
def update_weights(x_i, y_i, w):
    w[0] += x_i[0]*y_i
    w[1] += x_i[1]*y_i
    w[2] += x_i[2]*y_i

def sign(x):
    res = np.zeros(x.shape)
    positive_idx = np.where(x>=0)
    negative_idx = np.where(x<0)

    res[positive_idx] = 1
    res[negative_idx] = -1

    return res

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
    w = np.zeros((3,1))

    while True:
        num_iterations += 1
        perceptron = X_new.dot(w)
        y_pred = sign(perceptron)
        misclassified_points = np.where(y[:, np.newaxis] != y_pred)[0]
        all_good = misclassified_points.size == 0

        if all_good:
            break

        misclassified_point = np.random.choice(misclassified_points)

        update_weights(X_new[misclassified_point], y[misclassified_point], w)

    
    # As your decision function is simply sign(w0 + w1*x1 +w2*x2) then the decision boundary equation is a line with canonical form: w0 + w1*x1 + w2*x2 = 0
    decision_boundary_m = -w[1] / w[2]
    decision_boundary_b = -w[0] / w[2]

    X_1 = X_new[:, 1]
    X_2 = decision_boundary_m * X_1 + decision_boundary_b    
    
    if should_plot:
        plt.plot(X_1, X_2 , 'r')

    return num_iterations, decision_boundary_m, decision_boundary_b

def main():
    num_experiments = 1000
    total_iterations = 0
    
    should_plot = False

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

    avg_iterations = total_iterations / num_experiments
    print(avg_iterations)
    plt.show()

if __name__ == '__main__':
    main()

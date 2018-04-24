import numpy as np
import matplotlib.pyplot as plt
from utils import create_points, get_rand_linear_function, evaluate, LinearPredictor

low_range = -1
high_range = 1

d = 2
       
# def update_weights(x_i, y_i, w):
#     w[0] += x_i[0]*y_i
#     w[1] += x_i[1]*y_i
#     w[2] += x_i[2]*y_i

# def sign(x):
#     res = np.zeros(x.shape)
#     positive_idx = np.where(x>=0)
#     negative_idx = np.where(x<0)

#     res[positive_idx] = 1
#     res[negative_idx] = -1

#     return res

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
        plt.plot(X_1, X_2 , 'r')

    return decision_boundary_m, decision_boundary_b

def main():
    num_experiments = 1000
    total_iterations = 0
    
    should_plot = False


    sum_avg_disagree = 0
    for _ in range(num_experiments):
        X, y, target_func_m, target_func_b = create_points(
            num_points=100,
            low_range=low_range,
            high_range=high_range,
            d=d,
        )
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

        pred_func_m, pred_func_b = linear_regression(X, y, should_plot)
        linear_predictor = LinearPredictor(
            pred_func_m=pred_func_m,
            pred_func_b=pred_func_b,
        )

        sum_avg_disagree += evaluate(
            1000,
            pred_func_m,
            pred_func_b,
            target_func_m,
            target_func_b,
            low_range=low_range,
            high_range=high_range,
            d=d,
        )
    # evaluate(
    #     1000,
    #     pred_func_m,
    #     pred_func_b,
    #     target_func_m,
    #     target_func_b,
    #     low_range=low_range,
    #     high_range=high_range,
    #     d=d,
    # )
    # plt.show()

    total_e_in_avg_disagree = sum_avg_disagree / num_experiments
    print('total_e_in_avg_disagree', total_e_in_avg_disagree)

if __name__ == '__main__':
    main()

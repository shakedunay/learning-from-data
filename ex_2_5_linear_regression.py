import numpy as np
import matplotlib.pyplot as plt
from utils import create_labeled_points, get_rand_linear_function, evaluate, LinearPredictor

low_range = -1
high_range = 1

d = 2

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


def run_experiment(num_points, should_plot):
    func_m, func_bias = get_rand_linear_function(
        low_range=low_range,
        high_range=high_range,
    )
    X, y = create_labeled_points(
        func_m=func_m,
        func_bias=func_bias,
        num_points=num_points,
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

    pred_func_m, pred_func_b = linear_regression(
        X, y, should_plot)

    linear_predictor = LinearPredictor(
        func_m=pred_func_m,
        func_b=pred_func_b,
    )

    y_pred_e_in = linear_predictor.predict(X)
    avg_e_in = evaluate(
        y_pred=y_pred_e_in,
        y_true=y,
    )

    X_out, y_out = create_labeled_points(
        func_m=func_m,
        func_bias=func_bias,
        num_points=1000,
        low_range=low_range,
        high_range=high_range,
        d=d,
    )

    y_pred_e_out = linear_predictor.predict(X_out)
    avg_e_out = evaluate(
        y_pred=y_pred_e_out,
        y_true=y_out,
    )

    return avg_e_in, avg_e_out

def main():
    num_experiments = 1000

    should_plot = False

    for num_points in (100, ):
        sum_avg_e_in = 0
        sum_avg_e_out = 0
        for _ in range(num_experiments):
            avg_e_in, avg_e_out = run_experiment(num_points, should_plot)
            sum_avg_e_in += avg_e_in
            sum_avg_e_out += avg_e_out

        print('N', num_points)
        total_avg_e_in = sum_avg_e_in / num_experiments
        print('total_avg_e_in', total_avg_e_in)
        total_avg_e_out = sum_avg_e_out / num_experiments
        print('total_avg_e_out', total_avg_e_out)
if __name__ == '__main__':
    main()

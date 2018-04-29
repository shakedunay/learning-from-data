import numpy as np
from utils import create_labeled_points, get_rand_linear_function, evaluate, LinearPredictor, linear_regression, plot_points

low_range = -1
high_range = 1

d = 2

np.random.seed(1234)

def run_experiment(num_points, should_plot):
    func_m, func_bias = get_rand_linear_function(
        low_range=low_range,
        high_range=high_range,
    )
    X_in, y_in = create_labeled_points(
        func_m=func_m,
        func_bias=func_bias,
        num_points=num_points,
        low_range=low_range,
        high_range=high_range,
        d=d,
    )

    if should_plot:
        plot_points(X_in, y_in, first_color='blue', second_color='red')

    pred_func_m, pred_func_b = linear_regression(
        X_in, y_in, should_plot)

    linear_predictor = LinearPredictor(
        func_m=pred_func_m,
        func_b=pred_func_b,
    )

    y_pred_e_in = linear_predictor.predict(X_in)
    avg_e_in = evaluate(
        y_pred=y_pred_e_in,
        y_true=y_in,
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
    num_experiments = 1
    should_plot = True

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

    plt.show()
if __name__ == '__main__':
    main()

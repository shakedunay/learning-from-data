import random
import numpy as np

# https://github.com/kirbs-/edX-Learning-From-Data-Solutions/blob/master/Homework_2/Python/hw02_by_franck%20dernoncourt.py

def slope(x1, y1, x2, y2):
    return (y2-y1) / (x2-x1)

def intercept(x1, y1, x2 ,y2):
    # select either y2 ot y1 below
    return y2 - slope(x1, y1, x2, y2)

def generate_random_line():
    x1, y1, x2, y2 = [random.uniform(-1, 1) for i in range(4)]
    return {
        'w0': -1 * intercept(x1, y1, x2, y2),
        'w1': -1 * slope(x1, y1, x2, y2),
        'w2': 1,
    }


def generate_points(num_of_points):
    return [generate_point() for i in range(num_of_points)]

def experiment():
    target_line = generate_random_line()
    train_points = generate_points(num_of_points)

def main():
    experiment()

if __name__ == '__main__':
    main()

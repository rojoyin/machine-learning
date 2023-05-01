import numpy as np


def get_gradients(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw = (f_wb - y[i]) * x[i]
        dj_db = f_wb - y[i]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def run_gradient_descent(x, y, w, b, alpha, max_iterations):
    for i in range(max_iterations):
        dj_dw, dj_db = get_gradients(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

    return w, b


if __name__ == "__main__":
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])
    w_0 = 0
    b_0 = 0
    alpha_tmp = 1.0e-2
    max_number_of_iterations = 1000
    w, b = run_gradient_descent(x_train, y_train, w_0, b_0, alpha_tmp, max_number_of_iterations)
    print(f"w: {w}, b: {b}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data['distance'].values, data['elevation'].values


def lagrange_interpolation(x, y, x_new):
    n = len(x)
    m = len(x_new)
    y_interp = np.zeros(m)

    for i in range(m):
        p = 0
        for j in range(n):
            L = 1
            for k in range(n):
                if k != j:
                    L *= (x_new[i] - x[k]) / (x[j] - x[k])
            p += y[j] * L
        y_interp[i] = p

    return y_interp


def cubic_spline_interpolation(x, y, x_new):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * (delta[i] - delta[i - 1])

    c = np.linalg.solve(A, b)
    d = np.zeros(n - 1)
    b = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = delta[i] - h[i] * (2 * c[i] + c[i + 1]) / 3

    m = len(x_new)
    y_interp = np.zeros(m)

    for i in range(m):
        idx = np.searchsorted(x, x_new[i]) - 1
        idx = max(0, min(n - 2, idx))
        dx = x_new[i] - x[idx]
        y_interp[i] = y[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3

    return y_interp


def main():
    pass


if __name__ == '__main__':
    main()

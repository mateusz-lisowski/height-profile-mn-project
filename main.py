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


def main():
    pass


if __name__ == '__main__':
    main()

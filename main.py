import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data['distance'].values, data['elevation'].values


def main():
    pass


if __name__ == '__main__':
    main()

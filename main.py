"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: driver for the neural network file
"""

import numpy as np
import pandas as pd
import matplotlib as plot
from Neuron import Neuron

def load_data(path: str):
    """Loads csv data into a 2d numpy ndarray

    Args:
        path (str): file path of csv

    Returns:
        _type_: numpy array of numpy array's of extracted values
    """
    return pd.read_csv(path).values
    
    
    


def main():
    arr = load_data(".\\data\\mnist_test.csv")[0]
    print(arr)
    print(arr.shape)
    np.savetxt('test.csv', arr, fmt='%d', delimiter=',', newline=",")

if __name__ == "__main__":
    main()

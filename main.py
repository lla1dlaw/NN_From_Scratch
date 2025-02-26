"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: driver for the neural network file
"""

import numpy as np
import pandas as pd
import matplotlib as plot

def load_data(path: str):
    """Loads csv data into a 2d numpy ndarray

    Args:
        path (str): file path of csv

    Returns:
        _type_: numpy array of numpy array's of extracted values
    """
    return pd.read_csv(path).values
    

def main():
    train_data: np.ndarray = load_data(".\\data\\mnist_train.csv")
    test_data: np.ndarray = load_data(".\\data\\mnist_test.csv")

    print("Successfully Loaded Data.")
    print(f"Training Data Shape: {train_data.shape}")
    print(f"Testing Data Shape: {test_data.shape}")




if __name__ == "__main__":
    main()

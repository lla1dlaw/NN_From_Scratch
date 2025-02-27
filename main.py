"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Driver for MNist prediction using Neural networks
"""

import numpy as np
import pandas as pd
import matplotlib as plot
from Network import Network
import tensorflow as tf

def load_data(path: str):
    """Loads csv data into a 2d numpy ndarray

    Args:
        path (str): file path of csv

    Returns:
        _type_: numpy array of numpy array's of extracted values
    """
    return pd.read_csv(path).values


def train_network():
    ...


    

def main():
    # load data
    (train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = tf.keras.datasets.mnist.load_data()
    train_images = np.array(train_images_raw)
    train_labels = np.array(train_labels_raw)

    test_images = np.array(test_images_raw)
    test_labels = np.array(test_labels_raw)

    print("Successfully Loaded Data.\n")
    print(f"Train Images Shape: {train_images.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")
    print(f"Test Images Shape: {test_images.shape}")
    print(f"Test Labels Shape: {test_labels.shape}")


    






if __name__ == "__main__":
    main()

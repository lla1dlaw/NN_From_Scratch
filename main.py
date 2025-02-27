"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Driver for MNist prediction using Neural networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppresses warning from Tensor Flow
import tensorflow as tf
from Network import Network

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
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # labels are the second item in each pair
    training_pairs = list(zip(train_images, train_labels))
    testing_pairs = list(zip(test_images, test_labels))

    # define network
    input_size = training_pairs[0][0].size # should be 784 for 28 * 28 images in mnist
    num_outputs = 10 # 10 possible outputs 
    network_dimensions = [5]*3 # 5x3 symmetrical network

    net = Network(input_size, num_outputs, network_dimensions)

    
    









if __name__ == "__main__":
    main()

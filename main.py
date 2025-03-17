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



def show_number(image: np.ndarray, predicted_label: int, actual_label: int) -> None:
    """Displays a graphical representation of the supplied image with the associated labels.

    Args:
        image (np.ndarray): 2d array of pixels to display.
        predicted_label (int): Label that the network assigns to the image.
        actual_label (int): The true label for the image.
    """
    plt.title(f"Predicted Label: {predicted_label}\nTrue Label: {actual_label}")
    plt.imshow(image, cmap='gray')
    plt.show()
    

def main():
    # load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # labels are the second item in each pair
    training_pairs = list(zip(train_images, train_labels))
    testing_pairs = list(zip(test_images, test_labels))

    # network hyperparameters
    input_size = training_pairs[0][0].size # should be 784 for 28 * 28 images in mnist
    num_outputs = 10 # 10 possible outputs 
    network_dimensions = [10]*3 # 10x3 symmetrical network
    network_dimensions.append(num_outputs)

    # initialize network
    net = Network(input_size, num_outputs, network_dimensions)
    net.load_weights(".\\params")
    net.load_biases(".\\params")
    print("Network successfully initialized.")

    # evaluate network
    print("Evaluating network...")
    correct = 0
    for image, label in testing_pairs:
        prediction = net.forward(image.flatten())
        if prediction == label:
            correct += 1

    print(f"Network Accuracy: {correct/len(testing_pairs):.5f}")

if __name__ == "__main__":
    main()

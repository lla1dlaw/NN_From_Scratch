"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Driver for MNist prediction using Neural networks
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppresses warning from Tensor Flow
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Network import Network
from Loss import CrossEntropyLoss
from Optimizer import SGD



def load_data(path: str):
    """Loads csv data into a 2d numpy ndarray

    Args:
        path (str): file path of csv

    Returns:
        _type_: numpy array of numpy array's of extracted values
    """
    return pd.read_csv(path).values


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

def evaluate_network(net: np.ndarray, data: list[tuple[list[int], int]]) -> float:
    """Evaluates the network's performance on the supplied data.
    The data is expected to be a list of tuples with the first element being a 2d array of pixels and the second element being the label.

    Args:
        net (Network): Network being evaluated.
        data (list of tuples): list of pairs of input data and labels. 

    Returns:
        float: Accuracy of the network on the scale 0-1.
    """

    images = np.array([pair[0].flatten() for pair in data])
    labels = np.array([pair[1] for pair in data])

    predctions = net.forward(images)
    predicted_labels = np.argmax(predctions, axis=1)

    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    return np.mean(predicted_labels == labels)

def load_data():
    """Loads and returns mnist data

    Returns:
        _type_: _description_
    """
    print("\nLoading data...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_shape = train_images.shape
    test_shape = test_images.shape
    # flatten images
    train_images = train_images.reshape(train_shape[0], -1)
    test_images = test_images.reshape(test_shape[0], -1)
    training_pairs, testing_pairs = list(zip(train_images, train_labels)), list(zip(test_images, test_labels))
    print("\nData Loaded.")
    return training_pairs, testing_pairs

    
def main():
    # training parameters
    num_epochs = 10
    batch_size = 100

    # optimizer parameters
    initial_learning_rate = 1.0
    lr_decay = 0.001
    lr_decay_step = 1
    momentum = 0.9

    training_pairs, testing_pairs = load_data()

    # network hyperparameters
    input_size = training_pairs[0][0].size # should be 784 for 28 * 28 images in mnist
    num_outputs = 10 # 10 possible outputs
    network_dimensions = [10]*3 # 10x3 symmetrical hidden layers
    network_dimensions.append(num_outputs)

    # initialize network
    net = Network(input_size, num_outputs, network_dimensions)
    print("\nNetwork initialized.")
    print(f"\nNetwork Shape:\n{net}")

    # initialize optimizer
    optimizer = SGD
    optimizer.init(
        learning_rate=initial_learning_rate,
        decay_rate=lr_decay,
        decay_step=lr_decay_step,
        momentum=momentum
        )

    print("\nTraining Network...")
    # train network
    net.train(
        training_pairs=training_pairs, 
        epochs=num_epochs, 
        optimizer=optimizer, 
        batch_size=batch_size, 
        scramble_data=True
        )


    

   
    
    

if __name__ == "__main__":
    main()

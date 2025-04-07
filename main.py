"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Driver for MNist prediction using Neural networks
"""
import pretty_errors

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppresses warning from Tensor Flow
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Network import Network
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

def evaluate_network(net: Network, test_imgs: np.ndarray, test_labels: np.ndarray) -> float:
    """Evaluates the network's performance on the supplied data.
    The data is expected to be a list of tuples with the first element being a 2d array of pixels and the second element being the label.

    Args:
        net (Network): Network being evaluated.
        data (list of tuples): list of pairs of input data and labels. 

    Returns:
        float: Accuracy of the network on the scale 0-1.
    """

    predictions, loss = net.forward(test_imgs, test_labels)
    predicted_labels = np.argmax(predictions, axis=1)

    test_labels = np.argmax(test_labels, axis=1) if len(test_labels.shape) == 2 else test_labels        
    return np.mean(predicted_labels == test_labels), loss # return mean accuracy

def load_data():
    """Loads and returns mnist data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: training and testing images and labels as numpy arrays
    """
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # flatten images
    train_shape = train_images.shape
    test_shape = test_images.shape
    train_images = train_images.reshape(train_shape[0], -1)
    test_images = test_images.reshape(test_shape[0], -1)
    
    return train_images, train_labels, test_images, test_labels


def main():
    # training parameters
    num_epochs = 100
    batch_size = 50

    # optimizer parameters
    initial_learning_rate = 0.01
    lr_decay = 0.001
    lr_decay_step = 1
    momentum = 0.9

    print("\nLoading data...")
    training_images, training_labels, testing_images, testing_labels = load_data()
    print("Data Loaded.")


    # network hyperparameters
    input_size = training_images[0].size # should be 784 for 28 * 28 images in mnist
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

    # train network
    print("\nTraining Network...")
    accuracy_values, loss_values = net.train(
        training_images=training_images, 
        training_labels=training_labels,
        epochs=num_epochs, 
        optimizer=optimizer, 
        batch_size=batch_size, 
        scramble_data=True,
        epoch_print_step=5
        )
    print("\nNetwork Trained.")

    plt.plot(loss_values)
    plt.plot(accuracy_values)
    plt.show()

    # evaluate network
    print("\nEvaluating Network...")
    accuracy, loss = evaluate_network(net, testing_images, testing_labels)
    print(f"\nNetwork Test Accuracy: {accuracy*100:.5f}%\nNetwork Test Loss: {loss:.5f}")

if __name__ == "__main__":
    main()


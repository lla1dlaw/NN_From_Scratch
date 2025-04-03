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

import test
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppresses warning from Tensor Flow
import tensorflow as tf
from cv2 import resize
from Network import Network
from Loss import CrossEntropyLoss



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

    # correct = 0
    # for image, label in data:
    #     prediction = net.forward(np.array([image.flatten()]))
    #     if np.argmax(prediction) == label:
    #         correct += 1
    # print(f"Number of Samples: {len(data)}")
    # print(f"Prediction Shape: {prediction.shape}")
    # return correct/len(data)*100
    
def main():
    # load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # labels are the second item in each pair
    training_pairs = list(zip(train_images, train_labels))
    testing_pairs = list(zip(test_images, test_labels))

    # network hyperparameters
    input_size = training_pairs[0][0].size # should be 784 for 28 * 28 images in mnist
    num_outputs = 10 # 10 possible outputs
    network_dimensions = [10]*3 # 10x3 symmetrical hidden layers
    network_dimensions.append(num_outputs)

    # initialize network
    net = Network(input_size, num_outputs, network_dimensions)
    loss = CrossEntropyLoss()
    # net.load_weights(".\\torch-params")
    # net.load_biases(".\\torch-params")
    print("\nNetwork initialized.")
    print(f"\nNetwork Shape:\n{net}")
    
    # evaluate network
    # print("Evaluating network...")
    # accuracy = evaluate_network(net, testing_pairs)
    # print(f"Network Accuracy: {accuracy:.5f}%")

    flattened_images = np.array([image.flatten() for image in test_images[:11]])
    labels = np.array(test_labels[:11])

    print(f"Flattened Images Shape: {flattened_images.shape}")
    print(f"Labels Shape: {labels.shape}")

    preds = net.forward(flattened_images)
   
    loss_value = loss.calculate_loss(preds, labels)
    print(f"Loss Value: {loss_value}")

    accuracy = evaluate_network(net, testing_pairs)
    print(f"Network Accuracy: {accuracy*100:.5f}%")
    

if __name__ == "__main__":
    main()

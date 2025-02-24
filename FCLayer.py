"""
Filename: Layer.py
Author: Liam Laidlaw
Date: February 23, 2025
Purpose: Creates a collection of neurons stored inside a numpy array
"""

from Neuron import Neuron
import numpy as np

class FCLayer:
    def __init__(self, input_size: int, layer_width: int, starting_weights: np.array, starting_bias: int = 0) -> None:
        """Constructor

        Args:
            input_size (int): the size of the vectors inputted from the previous layer
            layer_width (int): the number of neurons in the layer
        """

        self.input_size = input_size
        self.layer_width = layer_width
        self.weights = starting_weights
        self.starting_bias = starting_bias

        # initialize each neuron in the layer with its own randomly sampled set of weights to apply to the outputs of the previous layer
        self.layer = [Neuron(weight_set, starting_bias) for weight_set in starting_weights]

    def forward(self, x: np.array) -> np.array:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): The inputted array of weighted sums from the previous layer

        Returns:
            np.array: The processed array of all the weighted sums produced from each neuron
        """
        return np.array([neuron.forward(x) for neuron in self.layer])
    

    ### fixx LATER
    def save_weights(self, path: str) -> None:
        """Saves the weight numpy array in csv format

        Args:
            path (str): path to save file
        """
        np.savetxt(path, self.weights(), fmt='%d', delimiter=',', newline=",")

        



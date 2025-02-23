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

        self.layer = [Neuron(starting_weights, starting_bias) for i in range(layer_width)]
        self.layer = np.array(self.layer)

    def forward(self, x: np.array) -> np.array:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): The inputted array of weighted sums from the previous layer

        Returns:
            np.array: The processed array of all the weighted sums produced from each neuron
        """
        return np.array([neuron.forward(x) for neuron in self.layer])

        



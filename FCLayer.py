"""
Filename: Layer.py
Author: Liam Laidlaw
Date: February 23, 2025
Purpose: Creates a collection of neurons stored inside a numpy array
"""

import numpy as np

class FCLayer:
    def __init__(self, input_size: int, layer_width: int, starting_bias: int = 0) -> None:
        """Constructor

        Args:
            input_size (int): the size of the vectors inputted from the previous layer
            layer_width (int): the number of neurons in the layer
            starting_bias (int, optional): The starting bias value for each of the neurons in the layer. Defaults to 0.

        """

        self.input_size = input_size
        self.layer_width = layer_width
        self.weights = None # must be explicitly initialized

        if input_size < 1: raise ValueError("input_size must be at least 1")
        if layer_width < 1: raise ValueError("layer_width must be at least 1")


        # one bias value for each neuron in the layer
        self.biases = np.array([0] * self.layer_width)

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): The inputted array of weighted sums from the previous layer

        Returns:
            np.array: The activation of each neuron in the layer
        """
        # compute the dot product and add bias
        return np.add(np.dot(input_vector, self.weights), self.biases)
    
    def set_weights(self) -> None:
        """Generates randomly sampled 2d numpy ndarray starting weights using 
        the He-et-al weight initialization method and stores it in self.weights

        Note: A different initialization should be used with hidden activation functions other than ReLU
        """
        self.weights = np.random.randn(self.layer_width, self.input_size) * np.sqrt(2/(self.input_size))

    def set_weights(self, weights: np.ndarray) -> None:
        """Sets weights for the layer.

        Args:
            weights (np.ndarray): 2d array of weight values. Each array is assigned to one unit.
        """

        self.weights = weights


    def set_biases(self, biases: np.ndarray) -> None:
        """Sets biases layer.

        Args:
            biases (np.ndarray): 1d array of bias values. Each array is assigned to one unit.
        """
        
        self.biases = biases

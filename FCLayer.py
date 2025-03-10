"""
Filename: Layer.py
Author: Liam Laidlaw
Date: February 23, 2025
Purpose: Creates a collection of neurons stored inside a numpy array
"""

import numpy as np

class FCLayer:
    def __init__(self, input_size: int, layer_width: int, starting_weights: np.ndarray = None, starting_bias: int = 0) -> None:
        """Constructor

        Args:
            input_size (int): the size of the vectors inputted from the previous layer
            layer_width (int): the number of neurons in the layer
            starting_weights (np.ndarray, optional): The starting weight matrix for the neurons in the layer. Defaults to None. 
            starting_bias (int, optional): The starting bias value for each of the neurons in the layer. Defaults to 0.

        """

        self.input_size = input_size
        self.layer_width = layer_width

        if input_size < 1: raise ValueError("input_size must be at least 1")
        if layer_width < 1: raise ValueError("layer_width must be at least 1")


        if starting_weights == None:
            self.initialize_starting_weights()
        else:
            self.weights = starting_weights

        self.biases = np.array([[0] for i in range(self.input_size)])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): The inputted array of weighted sums from the previous layer

        Returns:
            np.array: The processed array of all the weighted sums produced from each neuron
        """

        # append the value 1 to the end of the input array
        # This value is then multiplied by the bias in the weight matrix
        bias_placeholder = np.array([[1] for i in range(self.input_size)])
        input_vector = np.hstack((x, bias_placeholder))
        return np.dot(input_vector, np.hstack((self.weights, self.biases)))
    
    def initialize_starting_weights(self) -> None:
        """Generates randomly sampled 2d numpy ndarray starting weights using 
        the He-et-al weight initialization method and stores it in self.weights

        Note: A different initialization should be used with activation functions other than ReLU
        """
        self.weights = np.random.randn(self.input_size, self.layer_width) * np.sqrt(2/(self.input_size))

    def set_weights(self, weights: np.ndarray) -> None:
        """Sets weights for the layer.

        Args:
            weights (np.ndarray): 2d array of weight values. Each array is assigned to one unit.
        """
        
        self.weights = weights


    def set_biases(self, biases: np.ndarray) -> None:
        """Sets biases layer.

        Args:
            biases (np.ndarray): 2d array of bias values. Each array is assigned to one unit.
        """
        
        self.biases = biases

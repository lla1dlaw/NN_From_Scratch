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

        # bias weights are the last element of a given set of weights
        self.weights = np.hstack((self.weights, np.full(layer_width, starting_bias)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): The inputted array of weighted sums from the previous layer

        Returns:
            np.array: The processed array of all the weighted sums produced from each neuron
        """

        # append the value 1 to the end of the input array
        # This value is then multiplied by the bias in the weight matrix
        input_vector = np.hstack(x, np.full(1, 1)) 
        return np.dot(input_vector, self.weights)
    

    ### fix LATER
    def save_weights(self, path: str) -> None:
        """Saves the weight numpy array in csv format

        Args:
            path (str): path to save file
        """
        np.savetxt(path, self.weights(), fmt='%d', delimiter=',', newline=",")

    
    def initialize_starting_weights(self) -> None:
        """Generates randomly sampled 2d numpy ndarray starting weights using 
        the He-et-al weight initialization method and stores it in self.weights

        Note: A different initialization should be used with activation functions other than ReLU
        """
        self.weights = np.random.randn(self.input_size, self.layer_width) * np.sqrt(2/(self.input_size))


        



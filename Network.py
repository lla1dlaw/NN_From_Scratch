"""
Filename: Network.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: A neural network
"""

from FCLayer import FCLayer
import numpy as np
import Activation


class Network:
    def __init__(self, input_size: int, output_size: int, hidden_depth: int, symmetrical: bool = True, hidden_width: int = 10) -> None:
        """Constructor for Network Class

        Args:
            input_size (int): the dimensionality of input vector
            output_size (int): the dimensionality of output vector 
            hidden_depth (int): the depth of the network
            symmetrical (bool, optional): Defines whether the width of network hidden layers are to be the same size. Defaults to True.
            hidden_width (int, optional): Defines the width of each hidden layer in a symmetrical network. Defaults to 10.
            
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_depth = hidden_depth
        self.symmetrical = symmetrical
        self.hidden_width = hidden_width

        # starting values for every neuron
        starting_bias = 0
        starting_weights = []

        self.layers = []

        if self.symmetrical:
            # He-et-al weight initialization (used ONLY with ReLU activation function)
            # add 1 to the depth dimension so that a set of weights is created for the output layer
            first_layer_starting_weights = np.random.randn(input_size) * np.sqrt(2/(input_size))
            starting_weights = np.random.randn(hidden_width, hidden_depth + 1) * np.sqrt(2/(hidden_width))
            
            # first layer takes the input size as as the number of neurons
            self.layers.append(FCLayer(input_size, hidden_width, first_layer_starting_weights, starting_bias=starting_bias))
            for i in range(hidden_depth):
                self.layers.append(FCLayer(hidden_width, hidden_width, starting_weights[i], starting_bias=starting_bias))
            # output layer takes
            self.layers.append(FCLayer(hidden_width, output_size, starting_weights[-1], starting_bias=starting_bias))
        

    def load_weights(self, filename: str) -> None:
        """Loads and sets the model's weights and bias values

        Args:
            filename (str): The path to the file that contains the weights
        """
                
            



    def forward(self):
        ...
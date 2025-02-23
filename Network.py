"""
Filename: Network.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: A neural network
"""

from typing import Callable
import Neuron
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


        # input layer neurons are passive (don't change the input values)
        input_layer_weights = [1] * input_size 
        input_bias = 0

        # starting values for every neuron
        starting_bias = 0
        starting_weights = []

        layers = []

        if self.symmetrical:
            # He-et-al weight initialization (used ONLY with ReLU activation function)
            # add 1 to the depth dimension so that a set of weights is created for the output layer
            starting_weights = np.random.randn(hidden_width, hidden_depth+1) * np.sqrt(2/(hidden_width))
    
            # create input layer (one passive neuron for every feature in the input vector).
            input_layer = []
            for i in range(input_size):
                input_layer.append(Neuron(weights = 1, bias = 0))
            layers.append(np.array(input_layer)) # turn input layer into numpy array and add to layer list
                    
            # create first layer (weights are different dimensions than following layers)
            first_layer = [Neuron(np.random.randn(input_size)*np.sqrt(2/(hidden_width)), starting_bias) for i in range(hidden_width)]
            layers.append(np.array(first_layer))
                
            # create hidden layers
            for i in range(hidden_depth):
                hidden_layer = [Neuron(starting_weights[i][j], starting_bias) for j in range(hidden_width)]
                layers.append(hidden_layer)

            # create output layer
            output_layer = [Neuron(starting_weights[-1], starting_bias) for i in range(output_size)]
            layers.append(np.array(output_layer))
                
            



    def forward(self):
        ...
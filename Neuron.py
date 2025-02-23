"""
Filename: Neuron.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Defines a single neuron in a neural network
"""


import Activation
import numpy as np

class Neuron:
    def __init__(self, weights: np.array = None, bias: float = 1.0) -> None:
        """Constructor for a single Neuron

        Args:
            weights (np.array, optional): weights to be applied to each inputted value. Defaults to None.
            bias (float, optional): the scalar bias added to the dot product of the input and weights. Defaults to 1.0. 
            activation_function (Callable[[np.ndarray], float], optional): activation function applied to the biased weighted sum of the inputs. Defaults to activation.relu.
        """
        self.weights = weights
        self.bias = bias

    def forward(self, input) -> np.array:
        return np.dot(input, self.weights) + self.bias
    
    def set_weights(self, weights: np.array) -> None:
        self.weights = weights




    



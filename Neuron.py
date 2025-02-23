"""
Filename: Neuron.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Defines a single neuron in a neural network
"""

from typing import Callable
import activation
import numpy as np

class Neuron:
    def __init__(self, weights: np.array, bias: float, activation_function: Callable[[np.ndarray], float] = activation.relu):
        """Constructor for a single Neuron

        Args:
            weights (np.array): weights to be applied to each inputed value
            bias (float): the scalar bias added to the dot product of the input and weights
            activation_function (Callable[[np.ndarray], float], optional): activation function applied to the biased weighted sum of the inputs. Defaults to activation.relu.
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function


    def forward(self, input):
        return self.activation_function(np.dot(input, self.weights) + self.bias)




    



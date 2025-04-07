"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class Softmax:
    # empty intially, but will be changed to the output of the activation function
    def __init__(self):
        self.inputs = None
        self.input_gradients = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function (used after the output layer)

        Args:
            x (np.ndarray): The vector of values that needs to be converted into a probability distribution

        Returns:
            np.ndarray: Probability distribution representing the network's confidence in each output value
        """
        self.inputs = x.copy()
        # leverage numpy's vectorized operations to increase efficiency and speed of calculation
        stable_vector = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = stable_vector/(np.sum(stable_vector, axis=1, keepdims=True)) 
        return output 
        # second value is a placeholder 
        # (allows for easy iteration when loss is required in forward pass)
    
    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        """Defines a backwards pass through the softmax activation function. 
        (If softmax is applied to outputs of the network, use the combined loss and activation backwards method, 
        as it is more efficient)

        Args:
            derivatives (np.ndarray): The gradients from the previous layer in the backwards pass.

        Returns:
            np.ndarray: The gradients with respect to the forward pass inputs into the softmax function. 
        """
        self.input_gradients = np.empty_like(derivatives)

        for i, (output, deriv) in enumerate(zip(self.outputs, derivatives)):
            output = output.reshape(-1, 1)
            jacob_mat = np.diagflat(output) - np.dot(output, output.T)
            self.input_gradients[i] = np.dot(jacob_mat, deriv)

        return self.input_gradients

    
class ReLU:
    def __init__(self):
        self.input_gradients = None # gradients with respect to the forward pass inputs (intialized during the backwards pass)
        self.inputs = None # intialized in the forward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Relu activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
        """
        self.inputs = x.copy()
        return np.maximum(0, x)

    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        """Defines a backward pass through the ReLU function. 

        Args:
            derivatives (np.ndarray): The gradients of the previous layer in the backwards pass through the network. 

        Returns:
            np.ndarray: The gradients of the ReLU function with respect to the forward pass inputs.
        """
        self.input_gradients = derivatives.copy()
        # assigns gradient to 1 if the input is positive. 0 otherwise
        self.input_gradients[self.inputs <= 0] = 0
        return self.input_gradients
    
   


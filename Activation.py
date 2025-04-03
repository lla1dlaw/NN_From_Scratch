"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class Softmax:
    # empty intially, but will be changed to the output of the activation function
    inputs = None
    input_gradients = None


    @classmethod
    def forward(cls, x: np.ndarray) -> np.ndarray:
        """Softmax activation function (used after the output layer)

        Args:
            x (np.ndarray): The vector of values that needs to be converted into a probability distribution

        Returns:
            np.ndarray: Probability distribution representing the network's confidence in each output value
        """
        # leverage numpy's vectorized operations to increase efficiency and speed of calculation
        stable_vector = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = stable_vector/(np.sum(stable_vector, axis=1, keepdims=True)) 
        return output, 0 
        # second value is a placeholder 
        # (allows for easy iteration when loss is required in forward pass)
    
    @classmethod
    def backward(cls, derivatives: np.ndarray) -> np.ndarray:
        """Defines a backwards pass through the softmax activation function. 
        (If softmax is applied to outputs of the network, use the combined loss and activation backwards method, 
        as it is more efficient)

        Args:
            derivatives (np.ndarray): The gradients from the previous layer in the backwards pass.

        Returns:
            np.ndarray: The gradients with respect to the forward pass inputs into the softmax function. 
        """
        cls.input_gradients = np.empty_like(derivatives)

        for i, (output, deriv) in enumerate(zip(cls.outputs, derivatives)):
            output = output.reshape(-1, 1)
            jacob_mat = np.diagflat(output) - np.dot(output, output.T)
            cls.input_gradients[i] = np.dot(jacob_mat, deriv)

        return cls.input_gradients

    
class ReLU:
    input_gradients = None # gradients with respect to the forward pass inputs (intialized during the backwards pass)
    inputs = None # intialized in the forward pass

    @classmethod
    def forward(cls, x: np.ndarray) -> np.ndarray:
        """Relu activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
        """

        return np.maximize(0, x), 0 
        # second value is a placeholder 
        # (allows for easy iteration when loss is required in forward pass)

    @classmethod
    def backward(cls, derivatives: np.ndarray) -> np.ndarray:
        """Defines a backward pass through the ReLU function. 

        Args:
            derivatives (np.ndarray): The gradients of the previous layer in the backwards pass through the network. 

        Returns:
            np.ndarray: The gradients of the ReLU function with respect to the forward pass inputs.
        """
        cls.input_gradients = derivatives.copy()
        # assigns gradient to 1 if the input is positive. 0 otherwise
        cls.input_gradients[cls.inputs <= 0] = 0
        return cls.input_gradients
    
    # ------------------ activation functions for scalar inputs -------------------------

   


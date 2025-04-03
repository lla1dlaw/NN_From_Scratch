"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class Activation:
    # empty intially, but will be changed to the output of the activation function
    softmax_outputs = np.array([]) 

    @classmethod
    def softmax(cls, x: np.ndarray) -> np.ndarray:
        """Softmax activation function (used after the output layer)

        Args:
            x (np.ndarray): The vector of values that needs to be converted into a probability distribution

        Returns:
            np.ndarray: Probability distribution representing the network's confidence in each output value
        """
        # leverage numpy's vectorized operations to increase efficiency and speed of calculation
        stable_vector = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = stable_vector/(np.sum(stable_vector, axis=1, keepdims=True)) 
        cls.softmax_outputs = output
        return output
    

    @classmethod
    def relu(cls, x: np.ndarray) -> np.ndarray:
        """Relu activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
        """
        activation_function = np.vectorize(Activation.scalar_relu)
        return activation_function(x)

    
    # ------------------ activation functions for scalar inputs -------------------------

    @staticmethod 
    def scalar_relu(x: float) -> float:
        """Relu activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.maximum(0, x)


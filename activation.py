"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class activation:
    @staticmethod 
    def relu(x: float) -> float:
        """Relu activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.maximum(0, x)

    @staticmethod 
    def sigmoid(x: float) -> float:
        """sigmoid activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return 1/(1 + np.exp(-x))
    
    @staticmethod 
    def tanh(x: float) -> float:
        """hyperbolic tangent activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.tanh(x)
    
    @staticmethod 
    def softplus(x: float) -> float:
        """softplus activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.log(1 + np.exp(x))


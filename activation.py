"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class Activation:
    @staticmethod 
    def relu(x: np.array) -> np.array:
        """Relu activation over an inputted vector

        Args:
            x (np.array): array of values to activate

        Returns:
            np.array: array of activated values
        """
        activation_function = np.vectorize(Activation.scalar_relu)
        return activation_function(x)

    @staticmethod 
    def sigmoid(x: np.array) -> np.array:
        """Sigmoid activation over an inputted vector

        Args:
            x (np.array): array of values to activate

        Returns:
            np.array: array of activated values
        """
        activation_function = np.vectorize(Activation.scalar_sigmoid)
        return activation_function(x)
    
    @staticmethod 
    def tanh(x: np.array) -> np.array:
        """hyperbolic tangent activation over an inputted vector

        Args:
            x (np.array): array of values to activate

        Returns:
            np.array: array of activated values
        """
        activation_function = np.vectorize(Activation.scalar_tanh)
        return activation_function(x)
    
    @staticmethod 
    def softplus(x: np.array) -> np.array:
        """softplus activation over an inputted vector

        Args:
            x (np.array): array of values to activate

        Returns:
            np.array: array of activated values
        """
        activation_function = np.vectorize(Activation.scalar_softplus)
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

    @staticmethod 
    def scalar_sigmoid(x: float) -> float:
        """sigmoid activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return 1/(1 + np.exp(-x))
    
    @staticmethod 
    def scalar_tanh(x: float) -> float:
        """hyperbolic tangent activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.tanh(x)
    
    @staticmethod 
    def scalar_softplus(x: float) -> float:
        """softplus activation function

        Args:
            x (float): input into the activation function

        Returns:
            float: output of the activation function
        """
        return np.log(1 + np.exp(x))


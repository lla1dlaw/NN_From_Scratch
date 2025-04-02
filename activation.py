"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class Activation:

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
        return stable_vector/(np.sum(stable_vector, axis=1, keepdims=True)) 

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

    @classmethod 
    def sigmoid(cls, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
        """
        activation_function = np.vectorize(Activation.scalar_sigmoid)
        return activation_function(x)
    
    @classmethod 
    def tanh(cls, x: np.ndarray) -> np.ndarray:
        """hyperbolic tangent activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
        """
        activation_function = np.vectorize(Activation.scalar_tanh)
        return activation_function(x)
    
    @classmethod 
    def softplus(cls, x: np.ndarray) -> np.ndarray:
        """softplus activation over an inputted vector

        Args:
            x (np.ndarray): ndarray of values to activate

        Returns:
            np.ndarray: ndarray of activated values
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


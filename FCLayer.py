"""
Filename: Layer.py
Author: Liam Laidlaw
Date: February 23, 2025
Purpose: Creates a collection of neurons stored inside a numpy array
"""

from matplotlib.pylab import f
import numpy as np
from Activation import ReLU, Softmax

class FCLayer:
    def __init__(self, input_size: int, layer_width: int, activation, starting_bias: int = 0) -> None:
        """Constructor

        Args:
            input_size (int): the size of the vectors inputted from the previous layer
            layer_width (int): the number of neurons in the layer
            starting_bias (int, optional): The starting bias value for each of the neurons in the layer. Defaults to 0.

        """
        self.activation = activation

        self.input_size = input_size
        self.layer_width = layer_width

        self.inputs = None # stores the inputs into the layer for use in backwards pass

        # set default weight and bias values
        self.weights = 0.01 * np.random.randn(self.input_size, self.layer_width)
        self.biases = self.get_zeroed_biases()

        # create an empty gradient array for both weights and biases
        self.weight_gradients = np.empty_like(self.weights)
        self.bias_gradients = np.empty_like(self.biases)
        self.input_gradients = np.empty((input_size,))

        if input_size < 1: raise ValueError("input_size must be at least 1")
        if layer_width < 1: raise ValueError("layer_width must be at least 1")

    
    def get_random_weights(self) -> None:
        """Generates randomly sampled 2d numpy ndarray starting weights using
        the He-et-al weight initialization method and stores it in self.weights

        Note: A different initialization should be used with hidden activation functions other than ReLU
        """
        return np.random.randn(self.input_size, self.layer_width) * np.sqrt(2/(self.input_size))

    def get_zeroed_biases(self) -> None:
        """Generates a zeroed 1d numpy ndarray of biases
        """
        return np.zeros((1, self.layer_width), dtype=float)

    def set_weights(self, weights: np.ndarray) -> None:
        """Sets weights for the layer.

        Args:
            weights (np.ndarray): 2d array of weight values. Each array is assigned to one unit.
        """
        self.weights = weights

    def set_biases(self, biases: np.ndarray) -> None:
        """Sets biases for the layer.

        Args:
            biases (np.ndarray): 1d array of bias values. Each array is assigned to one unit.
        """
        self.biases = biases

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """Defines a forward pass through the layer and its neurons

        Args:
            x (np.array): 1d vector of weighted sums from the previous layer

        Returns:
            np.array: The activation of each neuron in the layer
        """

        self.inputs = input_vector
        
        if self.weights is None: raise ValueError("Weights have not been set")
        if self.biases is None: raise ValueError("Biases have not been set")
        # compute the dot product and add bias
        res = np.dot(input_vector, self.weights) + self.biases
        return res
    
    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        """Defines a backwards pass through each of the layer functions. 

        Args:
            derivatives (np.ndarray): Gradients from the previous layer in the backwards pass

        Returns:
            np.ndarray: Gradients for this layer
        """

        self.weight_gradients = np.dot(self.inputs.T, derivatives)
        self.bias_gradients = np.sum(derivatives, axis=0, keepdims=True)
        self.input_gradients = np.dot(derivatives, self.weights.T)

        

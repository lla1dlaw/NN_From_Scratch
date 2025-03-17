"""
Filename: Network.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: A neural network
"""

from FCLayer import FCLayer
from Activation import Activation
import numpy as np
import pandas as pd 


class Network:
    def __init__(self, input_size: int, output_size: int, hidden_widths: list[int]) -> None:
        """Constructor for Network Class

        Args:
            input_size (int): the dimensionality of input vector
            output_size (int): the dimensionality of output vector 
            hidden_widths (list[int]): describes the shape of the network. Each layer's width is represented by each integer value.

        Note:
            The depth of the network is derived from the length of the hidden_widths list
        """

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_widths = hidden_widths
        self.hidden_layers = []
            
        # first layer takes the input size as as the number of neurons
        self.hidden_layers.append(FCLayer(input_size=self.input_size, layer_width=self.hidden_widths[0]))

        # previous_layer_width is the input size for the current layer
        previous_layer_width = hidden_widths[0]
        for width in hidden_widths[1:-1]:
            self.hidden_layers.append(FCLayer(input_size=previous_layer_width, layer_width=width))
            previous_layer_width = width
        
        # output layer takes
        self.hidden_layers.append(FCLayer(input_size=self.hidden_widths[-1], layer_width=self.output_size))
        
    def load_weights(self, path: str) -> None:
        """Loads and sets the model's weight values

        Args:
            path (str): The path to the directory that contains the weights csv files
        """
        for i, layer in enumerate(self.hidden_layers):
            layer.set_weights(pd.read_csv(f"{path}\\layers.{i}.weight.csv", header=None).to_numpy())

    def load_biases(self, path: str) -> None:
        """Loads and sets the model's bias values

        Args:
            path (str): The path to the directory that contains the bias csv files
        """
        for i, layer in enumerate(self.hidden_layers):
            layer.set_biases(pd.read_csv(f"{path}\\layers.{i}.bias.csv", header=None).to_numpy())

    def set_parameters(self) -> None:
        """Sets the weights (randomly) and biases (zeroed) for each layer in the network"""
        for layer in self.hidden_layers:
            layer.set_weights()
            layer.set_biases()

    def save_weights(self, path: str) -> None:
        """Saves the current weights of a network to the specified path

        Args:
            path (str): The path to save the weights to
        """
        ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines a forward pass over the network

        Args:
            x (np.array): The inputted 1d feature vector

        Returns:
            np.ndarray: The output vector from the network. 
        """
        for i, layer in enumerate(self.hidden_layers[:-1]):
            x = layer.forward(x)
            x = Activation.relu(x)
            
        x = self.hidden_layers[-1].forward(x)
        return x


    def __str__(self):
        res = ""
        for i, layer in enumerate(self.hidden_layers):
            res += f"Layer {i}: {layer.weights.shape}\n"
        return res
"""
Filename: Network.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: A neural network
"""

from FCLayer import FCLayer
from Activation import Softmax, ReLU
from Optimizer import SGD
from Softmax_Crossentropy import Softmax_Categorical_CrossEntropy
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

        self.hidden_activation = ReLU
        self.activation_loss = Softmax_Categorical_CrossEntropy

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_widths = hidden_widths
        self.hidden_layers = []
            
        # first layer takes the input size as as the number of neurons
        self.hidden_layers.append(FCLayer(input_size=self.input_size, layer_width=self.hidden_widths[0]), self.hidden_activation)
        # previous_layer_width is the input size for the current layer
        previous_layer_width = hidden_widths[0]
        for width in hidden_widths[1:-1]:
            self.hidden_layers.append(FCLayer(input_size=previous_layer_width, layer_width=width), self.hidden_activation)
            previous_layer_width = width
        # output layer
        self.hidden_layers.append(FCLayer(input_size=self.hidden_widths[-1], layer_width=self.output_size), self.activation_loss)

        
    
    def train(self, data: np.ndarray, epochs, optimizer, batch_size: int = None) -> tuple[float, float]:
        """Training loop for the network

        Args:
            data (np.ndarray): _description_
            optimizer (_type_): _description_

        Returns:
            tuple[float, float]: _description_
        """
        if batch_size is None:
            batch_size = len(data)

        for epoch in range(epochs):

            self.backward() # propogate gradient

            if optimizer.decay_rate:
                optimizer.decay_learning_rate() # decay learning rate
            
            for layer in self.layers:
                optimizer.update_params(layer) # update parameters
                
            optimizer.step()
            

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Backpropogates gradients through the network.

        Args:
            predictions (np.ndarray): The output confidence predictions from the network. 
            labels (np.ndarray): Ground truth predictions of each class
        """     
        
        # propogate gradients backwards through the network layers and activation functions
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                activation_grad = layer.activation.backward(predictions, labels)
            else:
                activation_grad = layer.activation.backward(layer_grad)

            layer_grad = layer.backward(activation_grad)
    





        
    def load_weights(self, path: str) -> None:
        """Loads and sets the model's weight values

        Args:
            path (str): The path to the directory that contains the weights csv files
        """
        for i, layer in enumerate(self.hidden_layers):
            # layer.set_weights(pd.read_csv(f"{path}\\layers.{i}.weight.csv", header=None).to_numpy())
            with open(f"{path}\\layers.{i}.weight.csv", 'r') as file:
                lines = file.readlines()
                file.close()
            
            lines = [line.strip().split(',') for line in lines]
            weights = [[float(weight) for weight in line] for line in lines]
            layer_weights = np.array(weights)
            layer.set_weights(layer_weights)
                


    def load_biases(self, path: str) -> None:
        """Loads and sets the model's bias values

        Args:
            path (str): The path to the directory that contains the bias csv files
        """
        for i, layer in enumerate(self.hidden_layers):
            layer.set_biases(pd.read_csv(f"{path}\\layers.{i}.bias.csv", header=None).to_numpy())

    def save_weights(self, path: str) -> None:
        """Saves the current weights of a network to the specified path

        Args:
            path (str): The path to save the weights to
        """
        ...

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Defines a forward pass over the network

        Args:
            x (np.array): The inputted 1d feature vector

        Returns:
            tuple[np.ndarray, float]: (Network output predictions, average loss value)
        """
        for layer in self.hidden_layers:
            x = layer.forward(x)
            x = layer.activation.forward()

        return x


    def __str__(self):
        res = ""
        for i, layer in enumerate(self.hidden_layers):
            res += f"Layer {i}: {layer.weights.shape}\n"
        return res
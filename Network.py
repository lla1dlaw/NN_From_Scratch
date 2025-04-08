"""
Filename: Network.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: A neural network
"""

from FCLayer import FCLayer
from Activation import ReLU
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
            
        # previous_layer_width is the input size for the current layer
        previous_layer_width = input_size
        for width in hidden_widths[:-1]:
            self.hidden_layers.append(FCLayer(input_size=previous_layer_width, layer_width=width))
            self.hidden_layers.append(self.hidden_activation())
            previous_layer_width = width
        # output layer
        self.hidden_layers.append(FCLayer(input_size=self.hidden_widths[-1], layer_width=self.output_size))
        self.hidden_layers.append(self.activation_loss())
    
    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Defines a forward pass over the network

        Args:
            x (np.array): The inputted 1d feature vector

        Returns:
            tuple[np.ndarray, float]: (Network output predictions, average loss value)
        """

        for i, layer in enumerate(self.hidden_layers):
            # skip loss calculation unless on the last layer
            if i != len(self.hidden_layers) - 1: 
                x = layer.forward(x)
            else: 
                x, loss = layer.forward(x, y)
        return x, loss
    

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Backpropogates gradients through the network.

        Args:
            predictions (np.ndarray): The output confidence predictions from the network. 
            labels (np.ndarray): Ground truth predictions of each class
        """     
        
        # propogate gradients backwards through the network layers and activation functions
        gradient = None

        # iterate through layers in reverse order
        # layers activation functions are represented as "layers"
        for i, layer in enumerate(reversed(self.hidden_layers)):
            if i == 0:
                gradient = layer.backward(predictions, labels) # calculate gradient for the loss function
            else:
                gradient = layer.backward(gradient) # all other functions



    def shuffle_arrays(self, array1: np.ndarray, array2: np.ndarray) -> None:
        """Shuffles arrays in-place, in the same order, along axis=0.

        Args:
            arrays (list[np.ndarray]): List of NumPy arrays to shuffle.
            set_seed (float): Seed value if >= 0; otherwise, a random seed is used.
        """
        
        assert len(array1) == len(array2), "Arrays must be the same length"
        seed = np.random.randint(0, 2**(32 - 1) - 1) # generate a random seed 


        rstate = np.random.RandomState(seed)
        rstate.shuffle(array1)
        rstate = np.random.RandomState(seed)
        rstate.shuffle(array2)
        

    def train(self, training_images: np.ndarray, training_labels: np.ndarray, epochs, optimizer, batch_size: int = None, scramble_data: bool = False, epoch_print_step: int = 1) -> None:
        """Training loop for the network

        Args:
            training_images (np.ndarray): The training images to train the network on
            training_labels (np.ndarray): The training labels to train the network on
            epochs (int): The number of epochs to train the network for
            optimizer (Optimizer): The optimizer to use for training the network
            batch_size (int, optional): The number of samples to use in each batch. Defaults to None.
            scramble_data (bool, optional): Whether to shuffle the training data before each epoch. Defaults to False.
            epoch_print_step (int, optional): The number of epochs to wait before printing the training progress. Defaults to 1.
        Returns:
            tuple[np.ndarray, np.ndarray]: accuracy and loss values for each epoch
        """
        
        # copy training data
        train_imgs = training_images.copy()
        train_labels = training_labels.copy()

        # used to plot accuracy and loss values
        accuracy_values = []
        loss_values = []

        if batch_size is None:
            batch_size = len(training_data)
        
        num_batches = len(train_imgs)//batch_size # integer division to get clean subsets of the training data

        print(f"Number of batches: {num_batches}")
        
        for epoch in range(epochs):
            if scramble_data:# data needs to be scrambled before each epoch
                self.shuffle_arrays(train_imgs, train_labels) # shuffle data in place
            
            batched_imgs = np.split(train_imgs, num_batches)
            batched_labels = np.split(train_labels, num_batches)

            for img_batch, label_batch in zip(batched_imgs, batched_labels):
                img_batch = np.array(img_batch)
                label_batch = np.array(label_batch)
                predictions, loss = self.forward(img_batch, label_batch) # forward pass

                self.backward(predictions, label_batch) # propogate gradient

                if optimizer.decay_rate:
                    optimizer.decay_learning_rate() # decay learning rate
                
                for layer in self.hidden_layers:
                    optimizer.update_params(layer) # update network parameters
                    
                optimizer.step() # step optimizer

                one_hot_preds = np.argmax(predictions, axis=1)

                # ensure labels are one-hot encoded
                true_labels = np.argmax(label_batch, axis=1) if len(label_batch.shape) == 2 else label_batch 
                accuracy = np.mean(one_hot_preds==true_labels)

            if (epoch+1) % epoch_print_step == 0:
                print(f"Epoch: {epoch+1}, Acc: {accuracy*100:.5f}, Loss: {loss}, LR: {optimizer.learning_rate}")
            accuracy_values.append(accuracy) # save accuracy for plotting
            loss_values.append(loss) # save loss for plotting
        print(f"\nFinal Accuracy: {accuracy}\nFinal Loss: {loss}")
        return np.array(accuracy_values), np.array(loss_values) # return accuracy and loss values for plotting

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


    def __str__(self):
        res = ""
        for i, layer in enumerate(self.hidden_layers):
            res += f"Layer {i}: {layer.weights.shape}\n"
        return res

import numpy as np
from FCLayer import FCLayer

class SGD:
    learning_rate = 0.0 # must be explicitly set to intialize learning 

    @classmethod 
    def set_learning_rate(cls, learning_rate: float) -> None:
        """Sets the learning rate for the SGD optimizer

        Args:
            learning_rate (float): The new learning rate
        """
        cls.learning_rate = learning_rate

    @classmethod
    def update_params(cls, layer: FCLayer) -> None:
        """Updates the parameters of the layer using SGD

        Args:
            layer (FCLayer): The layer to update
        """
        layer.weights -= cls.learning_rate * layer.d_weights
        layer.biases -= cls.learning_rate * layer.d_biases
    

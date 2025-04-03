
from inspect import currentframe
import numpy as np
from FCLayer import FCLayer

class SGD:
    starting_learning_rate = None # initial learning rate 
    learning_rate = None # current calculated learning rate 
    decay_rate = None # must be explicitly set to intialize learning rate decay
    decay_step = None
    momentum = None
    num_iterations = 0 # number of iterations since the last decay

    @classmethod 
    def init(cls, learning_rate: float, decay_rate: float = None, decay_step: int = None, momentum: float = None) -> None:
        """Initializes the SGD optimizer with the given learning rate and decay rate.

        Args:
            learning_rate (float): The learning rate for the SGD optimizer
            decay_rate (float, optional): The decay rate for the SGD optimizer. Defaults to None.
            decay_step (int, optional): The decay step for the SGD optimizer. Defaults to None.
            momentum (float, optional): Learning rate momentum.
        """
        cls.starting_learning_rate = learning_rate
        cls.learning_rate = learning_rate
        cls.decay_rate = decay_rate
        cls.decay_step = decay_step
        cls.momentum = momentum


    @classmethod
    def decay_learning_rate(cls) -> None:
        """Decays the learning rate based on the decay rate, starting learning rate, and the number of iterations that optimizer has taken
        """
        if cls.learning_rate is None:
            current_frame = currentframe()
            raise ValueError(f"Learning rate must be set before calling method <{current_frame.f_code.co_name}>\nEnsure that init() has been called.")
        
        if cls.decay_rate is None or cls.decay_step is None:
            current_frame = currentframe()
            raise ValueError(f"Decay rate and decay step must be set before calling method <{current_frame.f_code.co_name}>.\nEnsure that the arguments decay_rate and decay_step are present in the inti() method.")
        
        cls.learning_rate = cls.starting_learning_rate * (1.0 / (1.0 + cls.decay_rate * cls.num_iterations))


    @classmethod
    def update_params(cls, layer: FCLayer) -> None:
        """Updates the parameters of the layer using SGD, with or without momentum

        Args:
            layer (FCLayer): The layer to update
        """

        if cls.learning_rate is None:
            current_frame = currentframe()
            raise ValueError(f"Learning rate must be set before calling method <{current_frame.f_code.co_name}>\nEnsure that init() has been called.")
        
        if cls.momentum: # if a momentum value has been specified
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weights_delta = cls.momentum * layer.weight_momentums - cls.learning_rate * layer.weight_gradients
            layer.weight_momentums = weights_delta
            biases_delta = cls.momentum * layer.bias_momentums - cls.learning_rate * layer.bias_gradients
            layer.bias_momentums = biases_delta

        else: # no momentum specified, use vanilla SGD
            weights_delta = -cls.learning_rate * layer.d_weights
            biases_delta = -cls.learning_rate * layer.d_biases

        # update weights and biases for that layer
        layer.weights += weights_delta
        layer.biases += biases_delta


    @classmethod
    def step(cls):
        """Increments number of iterations. Should be called once after all parameters are updated. 
        """
        cls.num_iterations += 1
    


from inspect import currentframe
import numpy as np
from FCLayer import FCLayer

class SGD:
    starting_learning_rate = None # initial learning rate 
    learning_rate = None # current calculated learning rate 
    decay_rate = None # must be explicitly set to intialize learning rate decay
    decay_step = None
    num_iterations = 0 # number of iterations since the last decay

    @classmethod 
    def init(cls, learning_rate: float, decay_rate: float = None, decay_step: int = None) -> None:
        """Initializes the SGD optimizer with the given learning rate and decay rate.

        Args:
            learning_rate (float): The learning rate for the SGD optimizer
            decay_rate (float, optional): The decay rate for the SGD optimizer. Defaults to None.
            decay_step (int, optional): The decay step for the SGD optimizer. Defaults to None.
        """
        cls.starting_learning_rate = learning_rate
        cls.learning_rate = learning_rate
        cls.decay_rate = decay_rate
        cls.decay_step = decay_step


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
        """Updates the parameters of the layer using SGD

        Args:
            layer (FCLayer): The layer to update
        """

        if cls.learning_rate is None:
            current_frame = currentframe()
            raise ValueError(f"Learning rate must be set before calling method <{current_frame.f_code.co_name}>\nEnsure that init() has been called.")
        
        layer.weights -= cls.learning_rate * layer.d_weights
        layer.biases -= cls.learning_rate * layer.d_biases


    @classmethod
    def step(cls):
        """Increments number of iterations. Should be called once after all parameters are updated. 
        """
        cls.num_iterations += 1
    

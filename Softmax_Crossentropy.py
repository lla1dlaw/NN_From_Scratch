
import numpy as np
from Activation import Activation
from Loss import CrossEntropyLoss

class Softmax_Categorical_CrossEntropy:
    activation = Activation.softmax()
    loss = CrossEntropyLoss()

    ouptut = np.array([]) # empty intially, but will be changed to the output of the activation function

    @classmethod
    def forward(cls, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the cross entropy loss between the output and target vectors

        Args:
            y_hat (np.ndarray): The output vector from the network
            y (np.ndarray): The target vector for the network

        Returns:
            np.ndarray: The loss vector
        """
        cls.output = Activation.softmax(y_hat)
        return cls.loss.calculate_loss(cls.output, y)
    
    @classmethod
    def backward(cls, derivatives: np.ndarray, y: np.ndarray) -> float:
        """Calculates the derivative of the loss function with respect to the output of the network

        Args:
            derivatives (np.ndarray): _description_
            y (np.ndarray): correct classifications

        Returns:
            float: the derivative of the loss function
        """
        num_samples = len(derivatives)

        # make sure that labels are one-hot
        if len(y.shape) == 1:
            y = np.argmax(y, axis=1)
        
        derivs = derivatives.copy()
        derivs[range(num_samples), y] -= 1
        derivs /= num_samples
        return derivs
    




import numpy as np
from Activation import Softmax
from Loss import CrossEntropyLoss

class Softmax_Categorical_CrossEntropy:
    def __init__(self) -> None:
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
        self.ouptut = None # None intially, but will be changed to the output of the activation function
        self.input_gradients = None # calculated in backwards pass

    
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the cross entropy loss between the output and target vectors

        Args:
            y_hat (np.ndarray): The output vector from the network
            y (np.ndarray): The target vector for the network

        Returns:
            np.ndarray: The loss vector
        """
        self.output = self.activation.forward(y_hat) # save output
        return self.output, self.loss.calculate_loss(y_hat, y) # return output for network and loss
    
    
    def backward(self, derivatives: np.ndarray, y: np.ndarray) -> float:
        """Calculates the derivative of the loss function with respect to the output of the network

        Args:
            derivatives (np.ndarray): Gradients from previous layer in the network
            y (np.ndarray): correct classifications

        Returns:
            float: gradients of the loss function
        """
        num_samples = len(derivatives)

        # make sure that labels are discrete if they are currently one-hot encoded
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        derivs = derivatives.copy()
        derivs[range(num_samples), y] -= 1
        derivs /= num_samples # normalize gradient
        self.input_gradients = derivs # save gradients for the next layer in the network
        return self.input_gradients
    



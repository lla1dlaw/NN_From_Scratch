import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates the cross entropy loss between the output and target vectors

        Args:
            y_hat (np.ndarray): The output vector from the network
            y (np.ndarray): The target vector for the network

        Returns:
            np.ndarray: The loss vector
        """


        # ensures non-zero or one values in y_hat for log calculation
        clipped_y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)

        if len(y.shape) == 1:
            num_samples = len(y_hat)
            correct_class_confidences = clipped_y_hat[range(num_samples), y]
        elif len(y.shape) == 2:
            correct_class_confidences = np.sum(clipped_y_hat*y, axis=1)
        else:
            raise ValueError("Invalid shape for target vector in loss calcyulation.")

        return -np.log(correct_class_confidences)
        

    def calculate_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the average cross entropy loss between the output and target vectors

        Args:
            y_hat (np.ndarray): The output vector from the network
            y (np.ndarray): The target vector for the network

        Returns:
            float: the average loss value
        """

        return np.mean(self.forward(y_hat, y))

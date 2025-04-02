import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Network import Network
from Loss import CrossEntropyLoss

nnfs.init()

def calculate_accuracy(y_hat, y):
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    return np.mean(y_hat == y)

def main():
    X, y = spiral_data(samples=100, classes=3)

    net = Network(input_size=2, output_size=3, hidden_widths=[3])
    loss = CrossEntropyLoss()
    predictions = net.forward(X)
    
    print(predictions[:5])
    loss_value = loss.calculate_loss(predictions, y)
    print(f"Loss Value: {loss_value}")

    predicted_classes = np.argmax(predictions, axis=1) # get the predicted class
    accuracy = calculate_accuracy(predicted_classes, y) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

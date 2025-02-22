"""
Filename: activation.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: Collection of activation functions for neural networks
"""

import numpy as np

class activation:
    def __init__():
        ...

    def relu(x: float) -> float:
        return np.maximum(0, x)

    def sigmoid(x: float) -> float:
        return 1/(1 + np.exp(-x))

    def tanh(x: float) -> float:
        return np.tanh(x)

    def softplus(x: float) -> float:
        return np.log(1 + np.exp(x))


"""
Filename: main.py
Author: Liam Laidlaw
Date: February 22, 2025
Purpose: driver for the neural network file
"""

import numpy as np
import pandas as pd
import matplotlib as plot
from Neuron import Neuron
 
def main():
    weights = np.array([2, 2, 2])
    bias = 1
    input = np.array([3, 4, 5])

    neuron = Neuron(weights=weights, bias=bias)

    print(f"Input Vector: {input}")
    print(f"Output Scalar: {neuron.forward(input)}")


if __name__ == "__main__":
    main()

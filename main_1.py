
import numpy as np

# Define the 1D vector and 2D matrix
v = np.array([1, 2, 3])
M = np.array([[4, 5, 6], [7, 8, 9]])

# Perform the dot product (using numpy.dot)
print(v.shape)
print(M.shape)
result = np.dot(M, v)

# Print the result
print(result)  # Output: [32 50]

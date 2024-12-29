import numpy as np

# Define constants
m1, d1, k1 = 4, 4, 4
m2, d2, k2 = 1, 1, 1

# Construct the matrix
matrix = np.array([
    [0, -(k2 + k1) / m1, 0, k2 / m1],
    [1, -(d2 + d1) / m1, 0, d2 / m1],
    [0, k2 / m2, 0, -k2 / m2],
    [0, d2 / m2, 1, -d2 / m2]
])

# Print the matrix
print("Matrix:")
print(matrix)

eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Print the eigenvalues
print("Eigenvalues:", eigenvalues)

# Print the eigenvectors (optional)
print("Eigenvectors:")
print(eigenvectors)
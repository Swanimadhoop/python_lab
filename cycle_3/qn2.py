import numpy as np

# Define the matrix X
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calculate the covariance matrix
cov_matrix = np.cov(X.T)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort the eigenvalues in decreasing order
sorted_indices = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the new principal component space
X_pca = X.dot(eigenvectors)

# Print the original matrix and the PCA-transformed matrix
print("Original matrix:")
print(X)
print("PCA-transformed matrix:")
print(X_pca)
     
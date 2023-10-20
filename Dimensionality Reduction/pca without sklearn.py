import numpy as np

# Generate random data
data = np.random.rand(100, 2)

# Step 1: Calculate the mean of the data
mean = np.mean(data, axis=0)

# Step 2: Center the data by subtracting the mean
centered_data = data - mean

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Step 4: Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Select the number of components (in this example, reduce to 1 component)
n_components = 1
selected_eigenvectors = eigenvectors[:, :n_components]

# Step 7: Project the data onto the selected eigenvectors
reduced_data = np.dot(centered_data, selected_eigenvectors)

# Print the reduced data
print("Reduced Data:")
print(reduced_data)
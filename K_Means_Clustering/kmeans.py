
import numpy as np
import matplotlib.pyplot as plt

# Generate random data points (you can replace this with your own dataset)
np.random.seed(0)
X = np.random.rand(100, 2)

# Number of clusters
k = 3

# Number of data points
m, n = X.shape

# Initialize cluster centroids randomly
centroids = X[np.random.choice(m, k, replace=False)]

# Maximum number of iterations
max_iterations = 100

# K-means algorithm
for _ in range(max_iterations):
    # Assign each data point to the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) #np.linalg.norm(...) computes the Euclidean distance between each data point and each centroid. The np.linalg.norm function calculates the Euclidean norm (distance) for each combination of data point and centroid. It returns an array of shape (m, k) where each element represents the distance between a data point and a centroid.


    labels = np.argmin(distances, axis=1)  #labels = np.argmin(distances, axis=1) assigns each data point to the cluster (centroid) with the shortest distance. np.argmin finds the index of the minimum distance for each data point along axis 1 (columns of the distances array), resulting in an array of cluster labels with shape (m,).



    # Update the centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    # Check for convergence
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Visualize the clustered data
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', label='Centroids')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(f'K-Means Clustering (k={k})')
plt.show()
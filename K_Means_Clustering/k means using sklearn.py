from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
data = np.random.rand(100, 2)

# Specify the number of clusters (K)
k = 3

# Create a KMeans instance and fit the data
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points with different colors for each cluster
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1,], label=f'Cluster {i + 1}')

# Plot the centroids as well
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
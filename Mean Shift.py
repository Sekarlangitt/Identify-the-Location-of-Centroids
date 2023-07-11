from typing import Tuple
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

def generate_sample_data(n_samples: int = 1000, n_features: int = 2, n_centers: int = 4, random_state: int = 42) -> np.ndarray:
    
    # Generate sample data
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=random_state)

    return data

def cluster_data(X: np.ndarray, quantile: float = 0.1) -> Tuple[int, np.ndarray, np.ndarray]:
    
    # Estimate the bandwidth of X
    bandwidth_X = estimate_bandwidth(X, quantile=quantile, n_samples=len(X))

    # Cluster data with MeanShift
    meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
    meanshift_model.fit(X)

    # Extract the centers of clusters
    cluster_centers = meanshift_model.cluster_centers_
    print('\nCenters of clusters:\n', cluster_centers)

    # Estimate the number of clusters
    labels = meanshift_model.labels_
    num_clusters = len(np.unique(labels))
    print("\nNumber of clusters in input data =", num_clusters)

    # Calculate the silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, labels)
    print("\nSilhouette average score for the clustering: ", silhouette_avg)

    return num_clusters, labels, cluster_centers

def plot_clusters(X: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray) -> None:
    
    # Plot the points and cluster centers
    plt.figure()
    markers = 'o*xvs'
    for i, marker in zip(range(len(cluster_centers)), markers):
        # Plot points that belong to the current cluster
        plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='blue')

        # Plot the cluster center
        cluster_center = cluster_centers[i]
        plt.plot(cluster_center[0], cluster_center[1], marker='o', 
                markerfacecolor='black', markeredgecolor='black', 
                markersize=15)

    plt.title('Clusters')
    plt.show()

# Generate sample data
data = generate_sample_data(n_samples=1000, n_features=2, n_centers=4, random_state=42)

# Cluster data using Mean Shift algorithm
num_clusters, labels, cluster_centers = cluster_data(X=data, quantile=0.1)

# Plot the resulting clusters
plot_clusters(X=data, labels=labels, cluster_centers=cluster_centers)
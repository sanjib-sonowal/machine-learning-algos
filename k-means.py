import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

sns.set()

X, y_tru = make_blobs(n_samples=3000, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()

# Assign four clusters
k_means = KMeans(n_clusters=4)
k_means.fit(X)
y_means = k_means.predict(X)


def find_clusters(X, n_clusters, r_seed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(r_seed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    _centers = X[i]
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, _centers)
        # 2b. Find new centers from mean of point
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c. Check for Convergence
        if np.all(_centers == new_centers):
            break
        _centers = new_centers
    return _centers, labels


# Data Visualization
v_centers, v_labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=y_means, s=50, cmap='viridis')
plt.scatter(v_centers[:, 0], v_centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

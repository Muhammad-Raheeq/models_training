# KMeans clustering example
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=13)
model = KMeans(n_clusters=3, random_state=13)
model.fit(X)
print('Kmeans cluster centers:', model.cluster_centers_)

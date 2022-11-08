from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Erstelle Daten
X, y = make_moons(n_samples=1000, noise=0.1 ,random_state=0)
# X, y = make_circles(n_samples=1000, noise=0.05 , factor = 0.4, random_state=0)

# # Erstelle Plot
plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Ground Truth")
plt.show()

kmeans = DBSCAN(eps=0.1, min_samples=4)
kmeans.fit(X)
y_clustering = kmeans.labels_

# Erstelle Plot vom Clustering Ergebnis
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y_clustering)
plt.title("DBSCAN Clustering")

plt.show()
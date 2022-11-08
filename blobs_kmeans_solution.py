from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Erstelle Daten
X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0, cluster_std=1)
# X, y = make_moons(n_samples=1000, noise=0.1 ,random_state=0)
# X, y = make_circles(n_samples=1000, noise=0.05 , factor = 0.4, random_state=0)

# Erstelle Plot
plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Ground Truth")

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)
y_clustering = kmeans.predict(X)

# Erstelle Plot vom Clustering Ergebnis
plt.figure(2)
plt.scatter(X[:,0], X[:,1], c=y_clustering)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], c="gray")
plt.title("KMeans Clustering")

# Berechne Werte für Ellbogen Plot
inertias = []
for i in range(2,15):
    kmeans_elbow = KMeans(n_clusters=i, random_state=0)
    kmeans_elbow.fit(X)
    inertias.append(kmeans_elbow.inertia_)

# Erstelle Elbow-Plot 
plt.figure(3)
plt.plot(range(2,15), inertias)
plt.title("Elbow Plot")

plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


# Ich generiere mich mit der scikit-learn-Funktion make_moons einen Datensatz. 
X, y = make_moons(n_samples=1000, noise = 0.1, random_state = None)


# # When the label y is 0, the class is represented with a blue square.
# # When the label y is 1, the class is represented with a green triangle.
# plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
# plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")


# # X contains two features, x1 and x2
# plt.xlabel(r"$x_1$", fontsize=20)
# plt.ylabel(r"$x_2$", fontsize=20)


# # Simplifying the plot by removing the axis scales.
# plt.xticks([])
# plt.yticks([])

# Displaying the plot.
plt.show()

# Ich erstelle ein Objekt und benutze Fit Methode
dbscan = DBSCAN(eps=0.5, min_samples= 5, metric= 'euclidean').fit(X)

# clustering = DBSCAN(eps=3, min_samples=2).fit(X)
dbscan.labels_

# clustering
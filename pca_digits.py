import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Lade Datensatz
X, y = datasets.load_digits(return_X_y=True)

print(X.shape)

# Erstelle ein PCA-Objekt
pca = PCA(n_components=64)

# Führe PCA durch
pca.fit(X[50:])
reduced_data = pca.transform(X)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(reduced_data[:50], y[:50])

# Aufgabe c) -->
print(clf.score(reduced_data[100:], y[100:]))
# conf = confusion_matrix(labels_test, pred_test)

# exit(0)
# Erstelle plot
fig = plt.figure(1)
# Die nächste Zeile nur für den 3D-Plot ausführen
ax  = fig.add_subplot(projection='3d')

# Erstelle je Label einen farbigen Plot
for label in np.unique(y):
    ids = np.where(y == label)
    ax.scatter(reduced_data[ids,0], reduced_data[ids,1], reduced_data[ids,2], label = label)
    # 2D-Plot
    #plt.scatter(reduced_data[ids,0], reduced_data[ids,1], label = label)

# Legende und Titel
plt.legend()
plt.title("Attributes of the digits dataset (PCA-reduced data)")

# Erstelle zweite Grafik 
plt.figure(2)
plt.plot(pca.explained_variance_ratio_)
plt.title("Explained variance ratio for each principal component")

# Kontrollrechnung
print("Summe der prozentualen erklärten Varianz:")
print(np.sum(pca.explained_variance_ratio_))

plt.show()
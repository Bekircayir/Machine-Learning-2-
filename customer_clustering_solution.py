import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Lade Datensatz
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))
df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))

np.random.seed(0)

# Zeige die ersten 5 Zeilen in der Konsole
print(df.head())

# Wähle Features
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features_real = df[col_names].values
print("num samples: ", features_real.shape[0])
genders = df['Gender'].values == 'Male'

# Aufgabe 1a)
# Mittelwert
for i in range(len(col_names)):
    print(col_names[i])
    print(" Mittelwert: ", np.mean(features_real[:,i]))
    print(" Standardabweichung: ", np.std(features_real[:,i]))
    print(" Kleinster Wert: ", np.min(features_real[:,i]))
    print(" Größter Wert: ", np.max(features_real[:,i]))

# Aufgabe 1b)
# Kovarianzmatrix
print("Kovarianzmatrix:")
features_transposed = np.transpose(features_real)
print(np.cov(features_transposed))

# Aufgabe 1c)
# Standardskalierung (Normalisierung)
# Variante 1:
scaler = StandardScaler()
scaler.fit(features_real)
features = scaler.transform(features_real)
# Variante 2:
mean = np.mean(features, axis=0)
std = np.std(features_real, axis=0)
features = (features_real - mean) / std

# Aufgabe 1d)
# Kovarianzmatrix nach Normalisierung
print("Kovarianzmatrix (nach Normalisierung):")
features_transposed = np.transpose(features)
print(np.cov(features_transposed))


# Aufgabe 2b)
# Elbow Plot & Average Silhouette Score
num_clusters = range(2,15)
inertias     = []
silhouettes  = []

for i in num_clusters:
    kmeans_test = KMeans(n_clusters=i)
    kmeans_test.fit(features)
    inertias.append(kmeans_test.inertia_)
    silhouettes.append(silhouette_score(features, kmeans_test.labels_))

plt.figure(2)
plt.plot(num_clusters, inertias)
plt.xlabel("Anzahl Cluster")
plt.ylabel("Gesamte Varianz")
plt.title("Elbow Plot")

plt.figure(3)
plt.plot(num_clusters, silhouettes)
plt.xlabel("Anzahl Cluster")
plt.ylabel("Durchschnittlicher Silhouette Score")
plt.title("Silhouette Plot")


# Aufgabe 2c)
# KMeans Clustering mit gewählter Cluster-Anzahl (6)
kmeans = KMeans(n_clusters=6)
kmeans.fit(features)
y_kmeans = kmeans.labels_
print("K-Means Silhouette Score: ", silhouette_score(features, kmeans.labels_))

# 3D Scatter Plot
fig = plt.figure(1)
ax  = fig.add_subplot(projection='3d')
ax.scatter(features_real[:,0], features_real[:,1], features_real[:,2], c=y_kmeans, cmap="Set3")
ax.set_xlabel(col_names[0])
ax.set_ylabel(col_names[1])
ax.set_zlabel(col_names[2])

# Aufgabe 2d)
for ci in range(6):
    selected_features = features_real[np.where(y_kmeans == ci)]
    selected_genders = genders[np.where(y_kmeans == ci)]

    means = np.mean(selected_features, axis=-1)
    std = np.std(selected_features, axis=-1) 
    male_percent = np.mean(selected_genders) * 100

    print("cluster ", ci)
    print("Gender: %.1f male / %.1f female" % (male_percent, 100-male_percent))
    print("Income: %.2f -- %.2f" % (means[0], std[0]))
    print("Score: %.2f -- %.2f" % (means[2], std[2]))
    print("Age: %.2f -- %.2f" % (means[1], std[1]))
    print("")

# Aufgabe 3b)
# Silhouette Scores für DBSCAN mit verschiedenen Werten
# für eps und min_samples berechnen
best_eps = 0
best_ms = 0
best_score = 0
best_clusters = 0
max_outlier = 0.3
for eps in np.arange(0.1, 2.0, 0.01):
    for min_samples in range(1,16):
        # Mit try/except werden ungültige Ergebnisse ignoriert
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(features)
            y_dbscan = dbscan.labels_

            # Outlier entfernen bevor Silhouette Score berechnet wird:
            feat = features[np.where(y_dbscan>-1)]
            y_dbscan = y_dbscan[np.where(y_dbscan>-1)]

            score = silhouette_score(feat, y_dbscan)

            # Ergebnis speichern, falls Silhouette Score verbessert wurde: 
            if score > best_score and np.mean(dbscan.labels_==-1) < max_outlier:
                best_score = score
                best_ms = min_samples
                best_eps = eps
                best_clusters = dbscan.labels_
        except:
            pass
print("DBSCAN bester Silhouette Score: ", best_score)
print("eps = %.4f | min_samples = %d" % (best_eps, best_ms))

fig = plt.figure(4)
ax  = fig.add_subplot(projection='3d')
ids = np.where(best_clusters > -1)
outlier = np.where(best_clusters == -1)
ax.scatter(features_real[ids,0], features_real[ids,1], features_real[ids,2], c=best_clusters[ids], cmap="Set3")
ax.scatter(features_real[outlier,0], features_real[outlier,1], features_real[outlier,2], c="black")
ax.set_xlabel(col_names[0])
ax.set_ylabel(col_names[1])
ax.set_zlabel(col_names[2])


plt.show()
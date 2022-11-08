import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Lade Datensatz
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))
df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mall_Customers.csv'))

# Zeige die ersten 5 Zeilen in der Konsole
print(df.head())

# Wähle Features
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features_real = df[col_names].values

X = df.iloc[:,3:6].values

print(df.describe())

age = df.iloc[:,2]
income = df.iloc[:,3]
score = df.iloc[:,4]

data = np.array([age, income, score])
print (np.cov(data, bias = True))



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 3, init='k-means++')
kmeans.fit(X)

clusters = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    clusters.append(kmeans.inertia_)

plt.plot(range(1,11),clusters)
plt.show()

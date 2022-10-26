from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np



digits = load_digits()

pca = PCA(n_components=3)
pca.fit(digits.data)

x = pca.transform(digits.data) 
y = digits.target

plt.scatter(x[:,0], x[:,1], c = y)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c = y)
plt.show()
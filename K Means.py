import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=10, centers=3, n_features=2,
                  random_state=0)
print(X.shape)

y

X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
                  random_state=0)
print(X.shape)

y

plt.scatter(X, y)
plt.show()
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)


digits = load_digits()


neigh.fit(digits.data[:50],digits.target[:50])
pred_test = neigh.predict(digits.data[50:])

print(neigh.score(digits.data[50:], digits.target[50:]))
conf = confusion_matrix(digits.target[50:], pred_test)
ergebnis = []

for i in range(2, 49):
    pca = PCA(n_components=i)
    pca.fit(digits.data[:50])
    x = pca.transform(digits.data) 
    neigh.fit(x[:50], digits.target[:50])
    pred_test_PCA = neigh.predict(x[50:])
    print(neigh.score(x[50:], digits.target[50:]), "Neue Score mit Transform Data n_components = ", i )
    conf = confusion_matrix(digits.target[50:], pred_test_PCA)
    ergebnis.append(neigh.score(x[50:], digits.target[50:]))
print(ergebnis, len(ergebnis))
x_point = np.arange(2,49 )
y_point = np.array(ergebnis)

plt.plot(x_point, y_point)
plt.show()

# y = digits.target
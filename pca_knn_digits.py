import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Lade Datensatz
X, y = datasets.load_digits(return_X_y=True)

# Train/Test Split
num_train = 50
X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# kNN ohne PCA
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)
print("Accuracy ohne PCA: %.4f" % test_accuracy)

# kNN mit PCA
# Teste mit verschiedenen Werten für n_components
accuracies = []
max_n = min(num_train, X.shape[1])
for n in range(2, max_n):
    pca = PCA(n_components=n)

    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    knn.fit(X_train_pca, y_train)

    test_accuracy = knn.score(X_test_pca, y_test)
    print("Accuracy mit PCA (n=%d): %.4f" % (n, test_accuracy))
    accuracies += [test_accuracy]

plt.plot(list(range(2,max_n)), accuracies)
plt.title("Accuracy vs. n_components")
plt.show()





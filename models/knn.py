import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        # Compute distances from x to all training samples at once
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # Get indices of the k nearest neighbors without sorting all distances
        k_indices = np.argpartition(distances, self.k)[:self.k]

        # Get labels of the nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        predictions = [self.predict_one(x) for x in X]
        return np.array(predictions)
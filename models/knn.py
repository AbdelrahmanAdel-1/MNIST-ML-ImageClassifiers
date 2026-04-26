import numpy as np


class KNNClassifier:
    def __init__(self, k=3, batch_size=500):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        KNN does not learn weights like logistic regression.
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _predict_batch(self, X_batch):
        """
        Predict labels for a batch of samples using vectorized distance calculation.
        """
        X_batch = np.asarray(X_batch)

        # Squared Euclidean distance:
        # ||x - train||^2 = ||x||^2 + ||train||^2 - 2*x.train
        X_batch_squared = np.sum(X_batch ** 2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train ** 2, axis=1)
        distances = X_batch_squared + X_train_squared - 2 * np.dot(X_batch, self.X_train.T)

        # Get indices of k nearest neighbors without sorting all distances
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]

        # Get labels of k nearest neighbors
        k_labels = self.y_train[k_indices]

        # Majority vote for binary classification
        predictions = []

        for labels in k_labels:
            counts = np.bincount(labels.astype(int))
            predictions.append(np.argmax(counts))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict labels for all samples using batches to avoid memory problems.
        """
        X = np.asarray(X)
        all_predictions = []

        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            batch_predictions = self._predict_batch(X_batch)
            all_predictions.extend(batch_predictions)

        return np.array(all_predictions)
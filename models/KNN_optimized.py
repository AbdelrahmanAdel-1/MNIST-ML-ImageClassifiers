import numpy as np


class KNNOptimized:
    def __init__(self, k=3, batch_size=500):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train)

    def _predict_batch(self, X_batch):
        X_batch = np.asarray(X_batch, dtype=np.float32)

        X_batch_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)
        X_train_sq = np.sum(self.X_train ** 2, axis=1)

        distances = X_batch_sq + X_train_sq - 2 * np.dot(X_batch, self.X_train.T)

        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        predictions = []

        for labels in k_labels:
            labels = labels.astype(int)
            counts = np.bincount(labels, minlength=10)
            predictions.append(np.argmax(counts))

        return np.array(predictions)

    def predict(self, X):
        predictions = []

        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            batch_pred = self._predict_batch(X_batch)
            predictions.extend(batch_pred)

        return np.array(predictions)
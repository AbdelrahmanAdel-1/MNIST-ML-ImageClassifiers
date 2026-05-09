import numpy as np


class KNNClassifier:
    def __init__(self, k=3, batch_size=500):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data.
        KNN does not learn weights.
        """
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y)

    def _predict_batch(self, X_batch):
        """
        Predict labels for batch using vectorized distances.
        """
        X_batch = np.asarray(X_batch, dtype=np.float32)

        X_batch_squared = np.sum(
            X_batch ** 2,
            axis=1,
            keepdims=True
        )

        X_train_squared = np.sum(
            self.X_train ** 2,
            axis=1
        )

        distances = (
            X_batch_squared
            + X_train_squared
            - 2 * np.dot(X_batch, self.X_train.T)
        )

        k_indices = np.argpartition(
            distances,
            self.k,
            axis=1
        )[:, :self.k]

        k_labels = self.y_train[k_indices]

        predictions = []

        for labels in k_labels:

            labels = labels.astype(int)

            counts = np.bincount(
                labels,
                minlength=10
            )

            predictions.append(np.argmax(counts))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict labels using batches
        to reduce memory usage.
        """
        X = np.asarray(X)

        predictions = []

        for start in range(
            0,
            len(X),
            self.batch_size
        ):

            end = start + self.batch_size

            X_batch = X[start:end]

            batch_predictions = self._predict_batch(
                X_batch
            )

            predictions.extend(batch_predictions)

        return np.array(predictions)
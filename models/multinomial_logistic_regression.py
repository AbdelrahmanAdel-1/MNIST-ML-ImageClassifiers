import numpy as np


class MultinomialLogisticRegression:

    def __init__(self, lr=0.01, epochs=1000, lambda_=0.01):

        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_

        self.W = None
        self.b = None

        self.loss_history = []
        self.val_loss_history = []

        self.train_acc_history = []
        self.val_acc_history = []

    def softmax(self, z):

        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, n_classes):

        one_hot_y = np.zeros((len(y), n_classes))

        one_hot_y[np.arange(len(y)), y] = 1

        return one_hot_y

    def compute_loss(self, y_true, y_pred, n_samples):

        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-15)) / n_samples

        l2_term = (self.lambda_ / (2 * n_samples)) * np.sum(self.W ** 2)

        return cross_entropy + l2_term

    def accuracy(self, y_true, y_pred):

        return np.mean(y_true == y_pred)

    def fit(self, X, y, X_val=None, y_val=None):

        n_samples, n_features = X.shape

        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        y_one_hot = self.one_hot(y, n_classes)

        for epoch in range(self.epochs):

            # Forward pass
            scores = np.dot(X, self.W) + self.b

            probs = self.softmax(scores)

            # Training loss
            train_loss = self.compute_loss(
                y_one_hot,
                probs,
                n_samples
            )

            self.loss_history.append(train_loss)

            # Gradients
            dW = (1 / n_samples) * np.dot(
                X.T,
                (probs - y_one_hot)
            )

            db = (1 / n_samples) * np.sum(
                probs - y_one_hot,
                axis=0,
                keepdims=True
            )

            # L2 regularization
            dW += (self.lambda_ / n_samples) * self.W

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            # Training accuracy
            train_preds = self.predict(X)

            train_acc = self.accuracy(y, train_preds)

            self.train_acc_history.append(train_acc)

            # Validation tracking
            if X_val is not None and y_val is not None:

                val_probs = self.predict_proba(X_val)

                y_val_one_hot = self.one_hot(y_val, n_classes)

                val_loss = self.compute_loss(
                    y_val_one_hot,
                    val_probs,
                    len(y_val)
                )

                self.val_loss_history.append(val_loss)

                val_preds = self.predict(X_val)

                val_acc = self.accuracy(y_val, val_preds)

                self.val_acc_history.append(val_acc)

    def predict_proba(self, X):

        scores = np.dot(X, self.W) + self.b

        return self.softmax(scores)

    def predict(self, X):

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)

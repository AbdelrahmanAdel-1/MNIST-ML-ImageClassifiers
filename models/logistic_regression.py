import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = 0
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred, n_samples):
        loss = -np.mean(
            y * np.log(y_pred + 1e-15) +
            (1 - y) * np.log(1 - y_pred + 1e-15)
        )
        # L2 regularization (exclude bias)
        loss += (self.lambda_ / (2 * n_samples)) * np.sum(self.w ** 2)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # stable initialization
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0

        for _ in range(self.epochs):
            # forward pass
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)

            # loss tracking
            loss = self.compute_loss(y, y_pred, n_samples)
            self.loss_history.append(loss)

            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # L2 regularization
            dw += (self.lambda_ / n_samples) * self.w

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # improved early stopping
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-7:
                    break

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def summary(self):
        print("Logistic Regression Model")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Lambda: {self.lambda_}")
        print(f"Final loss: {self.loss_history[-1] if self.loss_history else None}")

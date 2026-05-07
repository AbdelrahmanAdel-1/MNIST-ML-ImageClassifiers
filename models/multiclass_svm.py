import numpy as np
import matplotlib.pyplot as plt


def iterate_minibatches(X, y, batch_size=512, shuffle=True, seed=None):
    """
    Generate mini-batches from the full dataset.

    This function does not reduce the dataset.
    It only splits the full dataset into smaller batches for efficient training.

    Parameters:
        X          : feature matrix of shape (n_samples, n_features)
        y          : labels of shape (n_samples,)
        batch_size : number of samples per mini-batch
        shuffle    : whether to shuffle the data before batching
        seed       : random seed for reproducibility

    Yields:
        X_batch, y_batch
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]


class BinaryLinearSVM:
    """
    Binary Linear SVM trained using mini-batch vectorized gradient descent.

    Labels must be encoded as:
        +1 for the positive class
        -1 for the negative class

    Objective:
        J(w,b) = 0.5 * ||w||^2
                 + C * mean(max(0, 1 - y * (Xw + b)))

    Parameters:
        C             : regularization parameter
        learning_rate : gradient descent step size
        n_epochs      : number of passes over the full training set
        batch_size    : mini-batch size
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=20, batch_size=512):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the binary SVM.

        Parameters:
            X : array of shape (n_samples, n_features)
            y : array of shape (n_samples,), values must be +1 or -1

        Returns:
            self
        """
        n_samples, n_features = X.shape

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            epoch_losses = []

            for X_batch, y_batch in iterate_minibatches(
                X,
                y,
                batch_size=self.batch_size,
                shuffle=True,
                seed=epoch
            ):
                current_batch_size = X_batch.shape[0]

                scores = np.dot(X_batch, self.w) + self.b
                margins = y_batch * scores

                violations = margins < 1
                hinge_losses = np.maximum(0, 1 - margins)

                objective = (
                    0.5 * np.dot(self.w, self.w)
                    + self.C * np.mean(hinge_losses)
                )

                epoch_losses.append(objective)

                if np.any(violations):
                    dw = self.w - self.C * (
                        np.dot(X_batch[violations].T, y_batch[violations])
                        / current_batch_size
                    )

                    db = -self.C * (
                        np.sum(y_batch[violations])
                        / current_batch_size
                    )
                else:
                    dw = self.w
                    db = 0.0

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            mean_epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(mean_epoch_loss)

            print(
                f"Epoch {epoch + 1}/{self.n_epochs} "
                f"— Objective: {mean_epoch_loss:.4f}"
            )

        return self

    def decision_function(self, X):
        """
        Compute raw SVM scores.

        Formula:
            f(x) = w^T x + b

        Parameters:
            X : array of shape (n_samples, n_features)

        Returns:
            scores : array of shape (n_samples,)
        """
        X = X.astype(np.float32)
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Predict binary labels.

        Returns:
            +1 if score >= 0
            -1 otherwise
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


class OneVsRestLinearSVM:
    """
    Multiclass Linear SVM using the One-vs-Rest strategy.

    For K classes, this model trains K binary SVM classifiers.

    For each class k:
        +1 means the sample belongs to class k
        -1 means the sample belongs to any other class

    During prediction, the class with the highest decision score is selected.

    Parameters:
        C             : regularization parameter
        learning_rate : gradient descent step size
        n_epochs      : number of passes over the full training set
        batch_size    : mini-batch size
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=20, batch_size=512):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.classes_ = None
        self.classifiers = {}

    def fit(self, X, y):
        """
        Train one binary SVM classifier per class.

        Parameters:
            X : array of shape (n_samples, n_features)
            y : multiclass labels, for example 0 through 9

        Returns:
            self
        """
        self.classes_ = np.unique(y)
        self.classifiers = {}

        for class_label in self.classes_:
            print("=" * 70)
            print(f"Training Linear SVM for class {class_label} vs rest")
            print("=" * 70)

            y_binary = np.where(y == class_label, 1, -1)

            clf = BinaryLinearSVM(
                C=self.C,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size
            )

            clf.fit(X, y_binary)
            self.classifiers[class_label] = clf

        print("One-vs-Rest Linear SVM training completed.")
        return self

    def decision_function(self, X):
        """
        Compute decision scores for all classes.

        Parameters:
            X : array of shape (n_samples, n_features)

        Returns:
            scores : array of shape (n_samples, n_classes)

        Each column contains the scores from one class-specific classifier:
            s_k(x) = w_k^T x + b_k
        """
        if self.classes_ is None or len(self.classifiers) == 0:
            raise ValueError("Model must be fitted before calling decision_function().")

        scores = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)

        for idx, class_label in enumerate(self.classes_):
            clf = self.classifiers[class_label]
            scores[:, idx] = clf.decision_function(X)

        return scores

    def predict(self, X):
        """
        Predict multiclass labels.

        The predicted class is the one with the highest score:

            y_hat = argmax_k s_k(x)

        Parameters:
            X : array of shape (n_samples, n_features)

        Returns:
            predictions : array of shape (n_samples,)
        """
        scores = self.decision_function(X)
        best_indices = np.argmax(scores, axis=1)

        return self.classes_[best_indices]

    def plot_loss_curves(self, title="One-vs-Rest Linear SVM Loss Curves", save_path=None):
        """
        Plot the training objective curve for each binary classifier.

        Parameters:
            title     : plot title
            save_path : optional path to save the figure
        """
        if self.classes_ is None or len(self.classifiers) == 0:
            raise ValueError("Model must be fitted before plotting loss curves.")

        plt.figure(figsize=(10, 6))

        for class_label, clf in self.classifiers.items():
            plt.plot(
                clf.loss_history,
                marker="o",
                label=f"Class {class_label} vs rest"
            )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("SVM Objective")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

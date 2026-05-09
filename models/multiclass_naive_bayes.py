import numpy as np

class MulticlassNaiveBayes:
    def __init__(self, smoothing=1e-4):
        self.smoothing = smoothing 
        self.priors = None
        self.means = None
        self.vars = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i, :] = np.mean(X_c, axis=0)
            # Apply Smoothing Tuning here
            self.vars[i, :] = np.var(X_c, axis=0) + self.smoothing
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = - (x - mean)**2 / (2 * var)
        denominator = - 0.5 * np.log(2 * np.pi * var)
        return np.sum(numerator + denominator)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = [np.log(self.priors[i]) + self._calculate_likelihood(i, x) 
                      for i in range(len(self.classes))]
        return self.classes[np.argmax(posteriors)]

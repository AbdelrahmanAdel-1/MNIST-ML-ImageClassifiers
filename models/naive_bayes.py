
import numpy as np

class ManualNaiveBayes:
    def __init__(self):
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
            self.vars[i, :] = np.var(X_c, axis=0) + 1e-4 
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = - (x - mean)**2 / (2 * var)
        denominator = - 0.5 * np.log(2 * np.pi * var)
        return np.sum(numerator + denominator)

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = self._calculate_likelihood(i, x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

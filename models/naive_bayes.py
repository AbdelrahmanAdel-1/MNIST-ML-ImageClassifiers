import numpy as np

class ManualNaiveBayes:
    """
    A from-scratch implementation of the Gaussian Naive Bayes algorithm.
    This model assumes features (pixels) follow a normal distribution and 
    uses Bayes' Theorem to perform classification.
    """
    def __init__(self):
        # Placeholders for the parameters learned during training
        self.priors = None  # Class Prior probabilities P(y)
        self.means = None   # Mean pixel values (mu) for each class
        self.vars = None    # Pixel variances (sigma^2) for each class
        self.classes = None # Unique labels (e.g., [0, 1])

    def fit(self, X, y):
        """
        Training phase: Learns the statistical patterns of each class.
        Optimization: Uses Maximum Likelihood Estimation (MLE) to calculate mu and sigma.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameter matrices: [Number of Classes x Number of Pixels]
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            # Isolate samples belonging only to class 'c'
            X_c = X[y == c]

            # Learn the 'Average Image' template for this class
            self.means[i, :] = np.mean(X_c, axis=0)

            # Calculate pixel spread. We add a small epsilon (1e-4) to 
            # prevent division by zero for perfectly black pixels.
            self.vars[i, :] = np.var(X_c, axis=0) + 1e-4

            # Calculate class prevalence in the training set
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, class_idx, x):
        """
        Mathematical Engine: Implements the Gaussian Probability Density Function.
        Calculations are performed in Log-Space to ensure numerical stability 
        and prevent underflow when processing 784 features.
        """
        mean = self.means[class_idx]
        var = self.vars[class_idx]

        # Log-Gaussian Formula: -0.5 * log(2*pi*sigma^2) - ((x-mu)^2 / 2*sigma^2)
        numerator = - (x - mean)**2 / (2 * var)
        denominator = - 0.5 * np.log(2 * np.pi * var)

        # Sum the results for all pixels (Naive assumption of independence)
        return np.sum(numerator + denominator)

    def predict(self, X):
        """
        Inference phase: Loops through test samples to assign the most likely class.
        """
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """
        Decision Rule: Maximum A Posteriori (MAP).
        Combines learned Priors with calculated Likelihoods to find the highest score.
        """
        posteriors = []
        for i, c in enumerate(self.classes):
            # Final Score = log(P(y)) + log(P(x|y))
            prior = np.log(self.priors[i])
            likelihood = self._calculate_likelihood(i, x)
            posteriors.append(prior + likelihood)

        # Return the class label corresponding to the highest posterior probability
        return self.classes[np.argmax(posteriors)]

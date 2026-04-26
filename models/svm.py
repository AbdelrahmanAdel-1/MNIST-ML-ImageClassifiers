
import numpy as np

class LinearSVM:
    """
    Linear Support Vector Machine implemented from scratch using NumPy.
    Optimization is done using Gradient Descent with Hinge Loss.
    
    Parameters:
        C            : regularization parameter (default = 1.0)
        learning_rate: step size for gradient descent (default = 0.001)
        n_epochs     : number of training iterations (default = 1000)
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000):
        self.C             = C
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.w             = None
        self.b             = None
        self.loss_history  = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 1, 1, -1)
        self.w = np.zeros(n_features)
        self.b = 0.0
        for epoch in range(self.n_epochs):
            loss = 0
            dw   = np.zeros(n_features)
            db   = 0.0
            for i in range(n_samples):
                condition = y_[i] * (np.dot(X[i], self.w) + self.b)
                if condition >= 1:
                    dw += self.w
                else:
                    dw += self.w - self.C * y_[i] * X[i]
                    db += -self.C * y_[i]
                    loss += 1 - condition
            dw /= n_samples
            db /= n_samples
            total_loss = 0.5 * np.dot(self.w, self.w) + self.C * loss / n_samples
            self.loss_history.append(total_loss)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if (epoch + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{self.n_epochs} — Loss: {total_loss:.4f}")
        print("✅ Linear SVM training complete!")
    
    def predict(self, X):
        raw = np.dot(X, self.w) + self.b
        return np.where(raw >= 0, 1, 0)


class KernelSVM:
    """
    Kernel SVM using Nystroem Approximation + Linear SVM.
    Trains on the full dataset without memory issues.
    
    Parameters:
        C              : regularization parameter (default = 1.0)
        learning_rate  : step size for gradient descent (default = 0.001)
        n_epochs       : number of training iterations (default = 1000)
        degree         : polynomial kernel degree (default = 2)
        coef0          : polynomial kernel constant (default = 1)
        n_landmarks    : number of landmark points (default = 500)
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000,
                 degree=2, coef0=1, n_landmarks=500):
        self.C             = C
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.degree        = degree
        self.coef0         = coef0
        self.n_landmarks   = n_landmarks
        self.landmarks     = None
        self.w             = None
        self.b             = 0.0
        self.loss_history  = []
    
    def compute_kernel_features(self, X):
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        L_norm = self.landmarks / (np.linalg.norm(self.landmarks, axis=1, keepdims=True) + 1e-8)
        dot_product = np.dot(X_norm, L_norm.T)
        dot_product = np.clip(dot_product, -1, 1)
        K = (dot_product + self.coef0) ** self.degree
        return K
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        y_ = np.where(y == 1, 1, -1)
        print("   Sampling landmark points...")
        landmark_indices = np.random.choice(n_samples, self.n_landmarks, replace=False)
        self.landmarks   = X[landmark_indices]
        print("   Computing kernel features...")
        X_kernel = self.compute_kernel_features(X)
        print("   Training Linear SVM on kernel features...")
        self.w = np.zeros(self.n_landmarks)
        self.b = 0.0
        for epoch in range(self.n_epochs):
            scores    = np.dot(X_kernel, self.w) + self.b
            margins   = y_ * scores
            mask      = margins < 1
            loss      = np.sum(np.maximum(0, 1 - margins))
            dw        = self.w - self.C * np.dot(X_kernel[mask].T, y_[mask])
            db        = -self.C * np.sum(y_[mask])
            dw       /= n_samples
            db       /= n_samples
            total_loss = 0.5 * np.dot(self.w, self.w) + self.C * loss / n_samples
            self.loss_history.append(total_loss)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if (epoch + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{self.n_epochs} — Loss: {total_loss:.4f}")
        print("✅ Kernel SVM (Polynomial + Nystroem) training complete!")
    
    def predict(self, X):
        X_kernel    = self.compute_kernel_features(X)
        raw         = np.dot(X_kernel, self.w) + self.b
        return np.where(raw >= 0, 1, 0)

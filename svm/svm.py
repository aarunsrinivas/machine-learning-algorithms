import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class SVM:

    def __init__(self, C=1000, alpha=0.00005, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.C = C
        self.losses = []
        self.W = None
        self.b = None

    def fit(self, X, y):
        M, N = X.shape
        y = y.reshape((-1, 1))
        self.W = np.zeros((N, 1))
        self.b = np.zeros((1, 1))
        for _ in range(self.n_iterations):
            y_pred = X @ self.W + self.b
            w_grad = np.sum(np.where(
               (1 - y * y_pred) > 0,
               self.W[:, 0] - self.C * y * X,
               self.W[:, 0]
            ), axis=0).reshape((N, 1))
            b_grad = np.sum(np.where(
                (1 - y * y_pred) > 0,
                self.b[:, 0] - self.C * y,
                self.b[:, 0]
            ), axis=0).reshape((1, 1))
            hinge_loss = self.C / N * np.sum(np.maximum(0, 1 - y * y_pred))
            reg_loss = 1 / 2 * np.dot(self.W.T, self.W)
            loss = reg_loss + hinge_loss
            self.W -= (self.alpha / M) * w_grad
            self.b -= (self.alpha / M) * b_grad
            self.losses.append(loss.reshape(-1))
        return self

    def predict(self, X):
        y_pred = X @ self.W + self.b
        y_pred = np.where(y_pred > 0, 1, -1)
        return y_pred

X, y = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)
y[y == 0] = -1

model = SVM()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, alpha=0.05, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.losses = []
        self.W = None
        self.b = None

    def fit(self, X, y):
        M, N = X.shape
        y = y.reshape((-1, 1))
        self.W = np.zeros((N, 1))
        self.b = np.zeros((1, 1))
        for _ in range(self.n_iterations):
            z = X @ self.W + self.b
            y_pred = 1 / (1 + np.exp(-z))
            w_grad = X.T @ (y_pred - y)
            b_grad = np.sum(y_pred - y)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.W -= (self.alpha / M) * w_grad
            self.b -= (self.alpha / M) * b_grad
            self.losses.append(loss)
        return self

    def predict(self, X):
        z = X @ self.W + self.b
        y_pred = 1 / (1 + np.exp(-z))
        return np.around(y_pred)


X, y = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()





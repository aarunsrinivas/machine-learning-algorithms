import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, alpha=0.05, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.losses = []
        self.W = None
        self.b = None

    def fit(self, X, y):
        M, N = X.shape
        self.loss = []
        self.W = np.zeros((N, 1))
        self.b = np.zeros((1, 1))
        for _ in range(self.n_iterations):
            y_pred = X @ self.W + self.b
            w_grad = X.T @ (y_pred - y)
            b_grad = np.sum(y_pred - y)
            loss = np.sum((y_pred - y) ** 2 / (2 * M))
            self.W -= (self.alpha / M) * w_grad
            self.b -= (self.alpha / M) * b_grad
            self.losses.append(loss)
        return self

    def predict(self, X):
        y_pred = X @ self.W + self.b
        return y_pred


X = np.random.rand(100, 1)
y = 5 + 7 * X + np.random.rand(100, 1)
        
model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y)
plt.plot(X, model.W[0, 0] * X + model.b[0, 0])
plt.ylim((0, 15))
plt.show()




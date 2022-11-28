import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:

    def __init__(self):
        self.mu = []
        self.sigma = []
        self.p_y = []

    def fit(self, X, y):
        labels = np.unique(y)
        for label in labels:
            mu = np.mean(X[y == label], axis=0)
            sigma = np.std(X[y == label], axis=0)
            p_y = len(X[y == label]) / len(X)
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.p_y.append(p_y)
        return self
    
    def predict(self, X):
        predictions = []
        for data in X:
            prediction = []
            for i in range(len(self.p_y)):
                p = self.p_y[i]
                for j in range(len(data)):
                    p *= norm.pdf(data[j], self.mu[i][j], self.sigma[i][j])
                prediction.append(p)
            predictions.append(prediction)
        predictions = np.array(predictions)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred



X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = NaiveBayes()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
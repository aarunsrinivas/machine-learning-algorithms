import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RandomForest:

    def __init__(self, n_estimators=50, max_depth=None, max_features=0.7):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.decision_trees = [DecisionTreeClassifier(max_depth=max_depth, criterion='entropy') 
            for _ in range(n_estimators)]
        self.sample_indices = []
        self.feature_indices = []

    def _bootstrapping(self, num_training, num_features):
        row_idx = np.random.choice(num_training, num_training)
        col_idx = np.random.choice(num_features, int(num_features * self.max_features), replace=False)
        return row_idx, col_idx

    def bootstrapping(self, num_training, num_features):
        for _ in range(self.n_estimators):
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            self.sample_indices.append(row_idx)
            self.feature_indices.append(col_idx) 
        
    def fit(self, X, y):
        num_training, num_features = X.shape
        self.bootstrapping(num_training, num_features)
        for i in range(self.n_estimators):
            row_idx = self.sample_indices[i]
            col_idx = self.feature_indices[i]
            self.decision_trees[i].fit(X[row_idx, :][:, col_idx], y[row_idx])
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], 0))
        for i in range(self.n_estimators):
            p = self.decision_trees[i].predict(X[:, self.feature_indices[i]])
            predictions = np.append(predictions, p.reshape(-1, 1), axis=1)
        predictions = predictions.astype(np.int64)
        predictions = np.array([np.bincount(p).argmax() for p in predictions])
        return predictions


X, y = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForest()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

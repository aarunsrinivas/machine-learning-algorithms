import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree:

    def __init__(self, min_samples=2, max_depth=5):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, s):
        s = s.astype(np.int64)
        probabalities = np.bincount(s) / len(s)
        E = np.sum([-p * np.log2(p) for p in probabalities if p > 0])
        return E

    def information_gain(self, root, left, right):
        E_root = self.entropy(root)
        E_children = (len(left) * self.entropy(left) + len(right) * self.entropy(right)) / len(root)
        G = E_root - E_children
        return G

    def build_tree(self, X, depth=0):
        
        labels = X[:, -1]
        if np.all(labels == labels[0]):
            label = int(labels[0])
            return label

        if len(X) < self.min_samples or depth > self.max_depth:
            label = int(round(np.mean(labels)))
            return label
        
        node = None
        G_max = -1
        N = X.shape[1]
        for split_idx in range(N - 1):
            for split_value in np.unique(X[:, split_idx]):
                X_left = X[X[:, split_idx] <= split_value]
                X_right = X[X[:, split_idx] > split_value]
                G = self.information_gain(X[:, -1], X_left[:, -1], X_right[:, -1])
                if G > G_max:
                    node = {
                        'left': X_left,
                        'right': X_right,
                        'split_idx': split_idx,
                        'split_value': split_value
                    }
                    G_max = G
        
        return {
            'left': self.build_tree(node['left'], depth + 1),
            'right': self.build_tree(node['right'], depth + 1),
            'split_idx': node['split_idx'],
            'split_value': node['split_value']
        }

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        X = np.concatenate((X, y), axis=1)
        self.tree = self.build_tree(X)
        return self

    def r_predict(self, X, tree):
        if isinstance(tree, int):
            return tree
        if X[tree['split_idx']] <= tree['split_value']:
            return self.r_predict(X, tree['left'])
        else:
            return self.r_predict(X, tree['right'])

    def predict(self, X):
        return np.array([self.r_predict(data, self.tree) for data in X])


iris = load_iris()

X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTree()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))







import numpy as np
from collections import Counter


class decisionTree:
    # PM : Purity Measure; entropy by default
    def __init__(self, PM="entropy"):
        self.PM = PM
        self.tree = None

    def _probability(self, y):
        _, counts = Counter(y).keys(), Counter(y).values()
        return counts / len(y)

    def _purity(self, y):
        if self.PM == "entropy":
            return self._entropy(y)
        elif self.PM == "gini":
            return self._gini(y)
        else:
            raise ValueError(
                f"Invalid purity measure: {self.PM}. Use 'entropy' or 'gini'."
            )

    def _entropy(self, y):
        probabilities = self._probability(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _gini(self, y):
        probabilities = self._probability(y)
        return 1 - np.sum([p**2 for p in probabilities if p > 0])

    def _gain(self, X_column, y, threshold):
        parent_entropy = self._purity(y)
        left_indices = X_column < threshold
        right_indices = X_column >= threshold
        n, n_left, n_right = len(y), sum(left_indices), sum(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left / n) * self._purity(y[left_indices]) + (
            n_right / n
        ) * self._purity(y[right_indices])
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._gain(X[:, feature_index], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))

        if depth == self.max_depth or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None:
            return {'leaf': True, 'value': self._most_common_label(y)}

        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'leaf': False,
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

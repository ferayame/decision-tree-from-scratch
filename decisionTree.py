import numpy as np
from collections import Counter


class DecisionTree:
    # PM : Purity Measure; entropy by default
    def __init__(self, PM="entropy", max_depth=5):
        self.PM = PM
        self.max_depth = max_depth
        self.tree = None

    def _probability(self, y):
        counts = np.array(list(Counter(y).values()))
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
        n, n_left, n_right = len(y), np.sum(left_indices), np.sum(right_indices)

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
            thresholds = np.unique(X.iloc[:, feature_index])
            if len(thresholds) == 1:
                continue
            for threshold in thresholds:
                gain = self._gain(X.iloc[:, feature_index], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))

        if depth == self.max_depth or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None or threshold is None:
            return {"leaf": True, "value": self._most_common_label(y)}

        left_indices = X.iloc[:, feature_index] < threshold
        right_indices = X.iloc[:, feature_index] >= threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "leaf": False,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _most_common_label(self, y):
        label_counts = Counter(y)
        return label_counts.most_common(1)[0][0]

    def _predict_sample(self, x, tree):
        if tree["leaf"]:
            return tree["value"]

        feature_value = x[tree["feature_index"]]
        if feature_value < tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X.values])
    
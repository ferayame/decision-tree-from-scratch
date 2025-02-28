import numpy as np

def _best_split(self, X, y):
    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])  # Get unique values as thresholds
        for threshold in thresholds:
            left_indices = X[:, feature] < threshold
            right_indices = X[:, feature] >= threshold

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            gini = self._gini_impurity(y[left_indices], y[right_indices])
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold





import numpy as np

def residual_information(subset_distributions):
    """
    Calculate the residual information after a split using entropy.

    Parameters:
    subset_distributions (list of dicts): A list where each element is a dictionary
                                          representing the class distribution in a subset.
                                          Example: [{'A': 5, 'B': 2}, {'A': 1, 'B': 2}]

    Returns:
    float: The residual information.
    """
    total_samples = sum(sum(subset.values()) for subset in subset_distributions)
    residual_info = 0.0

    for subset in subset_distributions:
        subset_size = sum(subset.values())
        p_v = subset_size / total_samples  # Probability of the subset
        subset_entropy = 0.0

        for count in subset.values():
            p_c_given_v = count / subset_size  # Probability of class given the subset
            if p_c_given_v > 0:
                subset_entropy -= p_c_given_v * np.log2(p_c_given_v)

        residual_info += p_v * subset_entropy

    return residual_info



class decisionTree:
    # PM : Purity Measure; entropy by default
    
    def __init__(self, PM = 'entropy'):
        self.PM = PM
        self.tree = None


    # Purity Measures
    def _entropy(self, probabilities):
        entropy_value = 0
        for p in probabilities:
            if p != 0:  # Skip if p is 0
                entropy_value -= p * np.log2(p)
        return entropy_value
    
    def _gini(self, probabilities):
        gini_value = 0
        for p in probabilities:
            if p != 0:
                gini_value += p ** 2 
                
        return 1 - gini_value
        
    
    def _resInformation(self, probabilities, conditional_probabilities):
        pass
    
    def _gain(self, attribute):
        return self._entropy()
    
    def _split(self, feature):
        pass
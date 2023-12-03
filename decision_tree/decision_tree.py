import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class DecisionTree:
    
    def __init__(self, max_depth=None):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y, depth=0):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.tree = self._build_tree(X, y, depth)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._predict_row(self.tree, row))
        return pd.Series(predictions, index=X.index)
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        rules = []
        self._get_rules_recursive(self.tree, [], rules)
        return rules


    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or y.nunique() == 1:
            return y.mode().iloc[0]  # Return the most common class label

        if len(X.columns) == 0:
            return y.mode().iloc[0]  # Return the most common class label

        best_attribute, best_split = self._find_best_split(X, y)
        if best_attribute is None:
            return y.mode().iloc[0]  # Return the most common class label

        tree = {best_attribute: {}}
        for value, subset in best_split.items():
            tree[best_attribute][value] = self._build_tree(subset[0], subset[1], depth + 1)

        return tree
    

    def _find_best_split(self, X, y):
        entropy_before_split = self._entropy(y)
        best_info_gain = 0
        best_attribute = None
        best_split = None

        for attribute in X.columns:
            values = X[attribute].unique()
            subsets = {}
            for value in values:
                subsets[value] = (X[X[attribute] == value], y[X[attribute] == value])

            entropy_after_split = sum(len(subset[1]) / len(y) * self._entropy(subset[1]) for subset in subsets.values())
            info_gain = entropy_before_split - entropy_after_split

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
                best_split = subsets

        return best_attribute, best_split
    

    def _entropy(self, y):
        value_counts = y.value_counts()
        probabilities = value_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    

    def _predict_row(self, tree, row):
        if isinstance(tree, str):  # Leaf node
            return tree
        else:
            attribute = list(tree.keys())[0]
            value = row[attribute]
            if value in tree[attribute]:
                return self._predict_row(tree[attribute][value], row)
            else:
                return None  # Value not found in the tree


    def _get_rules_recursive(self, tree, rule, rules):
        if isinstance(tree, str):  # Leaf node
            rules.append((rule, tree))
        else:
            attribute = list(tree.keys())[0]
            for value in tree[attribute]:
                self._get_rules_recursive(tree[attribute][value], rule + [(attribute, value)], rules)


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


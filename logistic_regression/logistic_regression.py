import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, learning_rate=0.001, num_iterations=100000, batch_size=10, regularization_strength=0, scale=False, polynomial_features=False, degree=1):
        # Initialize hyperparameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.scale = scale
        self.polynomial_features = polynomial_features
        self.degree = degree
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """

        X = np.array(X)  # Convert X to a NumPy array
        y = np.array(y)  # Convert y to a NumPy array

        # Preprocess the data
        X = self.preprocess_data(X)
        
        m, n = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            # Shuffle the dataset and labels together
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, m, self.batch_size):
                # Select a batch of data
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Compute the linear combination of input and weights
                linear_combination = np.dot(X_batch, self.weights) + self.bias
                # Apply the sigmoid function to get probabilities
                predictions = sigmoid(linear_combination)
                
                # Compute the gradient of the loss function
                dw = (1/len(X_batch)) * (np.dot(X_batch.T, (predictions - y_batch)) + 2 * self.regularization_strength * self.weights)  # L2 regularization term added
                db = (1/len(X_batch)) * np.sum(predictions - y_batch)
                
                # Update the weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained. Please call fit() first.")
        

        X = np.array(X)  # Convert input to a NumPy array
        # Preprocess the test data using the same polynomial features
        X_preprocessed = self.preprocess_data(X)
        
        linear_combination = np.dot(X_preprocessed, self.weights) + self.bias
        predictions = sigmoid(linear_combination)
        return predictions
    
    def preprocess_data(self, X):
        # Implement polynomial feature transformation
        if self.polynomial_features:
            X = self.add_polynomial_features(X, self.degree)
        # Scaling
        if self.scale:
            X = self.scale_data(X)
        return X
    
    def scale_data(self, X):
        """
        Min-Max scaling for a feature matrix X.

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            X_scaled: Min-Max scaled feature matrix
        """
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X_scaled = (X - min_vals) / (max_vals - min_vals)
        return X_scaled


    def add_polynomial_features(self, X, degree=2):
        """
        Add polynomial features to the input matrix X.
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            degree (int): the degree of the polynomial features to add
        
        Returns:
            X_poly: the input matrix with polynomial features
        """
        if degree < 2:
            return X

        m, n = X.shape
        X_poly = X.copy()

        for d in range(2, degree + 1):
            for i in range(n):
                feature_i = X[:, i]
                new_feature = feature_i ** d
                X_poly = np.column_stack((X_poly, new_feature))

        return X_poly
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

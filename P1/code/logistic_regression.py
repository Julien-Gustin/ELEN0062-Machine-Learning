"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError(
                "This class is only dealing with binary " "classification problems"
            )

        w1 = 1  # init?
        w2 = 1

        self.theta_ = np.array([1, w1, w2])  # [bias, param1, param2]
        bias = np.ones(n_instances)  # [1, ..., 1]
        X_ = np.c_[
            bias, X
        ]  # [[1, x_0,1, x_0,2], ... [1, x_(n_instances-1),1 , x_(n_instances-1),2]]

        for _ in range(self.n_iter):  # do the gradiant descent
            sum_ = np.zeros(len(self.theta_))
            for xi, yi in zip(X_, y):  # iter for each instances
                sum_ += (1 / (1 + np.math.exp(-self.theta_.dot(xi))) - yi) * xi

            loss = 1 / n_instances * sum_  # compute the loss
            self.theta_ = self.theta_ - self.learning_rate * loss  # update parameters
            print(self.theta_)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        y = list()
        n_instances = X.shape[0]
        bias = np.ones(n_instances)
        X_ = np.c_[bias, X]
        for sample in X_:
            product = self.theta_.dot(sample)
            y.append(int(product >= 0))
        return np.array(y)
        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        p = list()
        n_instances = X.shape[0]
        bias = np.ones(n_instances)
        X_ = np.c_[bias, X]
        for sample in X_:
            probability = (1 / (1 + np.math.exp(-self.theta_.dot(sample))))
            p.append((probability, 1-probability))
        return np.array(p)
        pass


if __name__ == "__main__":
    X, y = make_unbalanced_dataset(1000, random_state=42)
    logistic_reg = LogisticRegressionClassifier()
    logistic_reg.fit(X, y)
    z = np.array([1.2, 1.3])
    print(logistic_reg.predict(X))
    print(logistic_reg.predict_proba(X))
    pass

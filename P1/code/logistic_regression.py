"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import prod

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


def split_train_test(
    train_size: int = 1000, samples: int = 3000, seed: int = 42
) -> tuple:
    """Return the train and test set

    Parameters
    ----------
    train_size : int 
                 Size of training set. Defaults to 1000.
    samples : int
              Samples' size. Defaults to 3000.
    seed : int
           The random state

    Returns:
        X_train: array of shape = [n_points, 2]
        X_test: array of shape = [n_points, 2]
        y_train: array of shape = [n_points]
        y_test: array of shape = [n_points]
    """
    X_full, y_full = make_unbalanced_dataset(samples, random_state=seed)

    X_train, X_test = X_full[:train_size], X_full[train_size:]
    y_train, y_test = y_full[:train_size], y_full[train_size:]
    return X_train, X_test, y_train, y_test


def sigmoid_function(x):
    """Apply the sigmoid function to `x`

    Parameters
    ----------
    x : A reel number
        The input of sigmoid function

    Returns:
        p : a reel number between 0 and 1
    """
    p = 1 / (1 + np.math.exp(-x))
    return p


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

        self.coef_ = np.ones(n_features)
        self.intercept_ = 1

        # do the gradiant descent
        for _ in range(self.n_iter):
            sum_coef = np.zeros(len(self.coef_))
            sum_intercept = 0

            # iter for each instances
            for x_i, y_i in zip(X, y):
                residual = sigmoid_function(self.coef_.dot(x_i) + self.intercept_) - y_i
                sum_coef += residual * x_i
                sum_intercept += residual

            # compute the loss
            loss_coef = 1 / n_instances * sum_coef
            loss_intercept = 1 / n_instances * sum_intercept

            # update parameters
            self.coef_ = self.coef_ - self.learning_rate * loss_coef
            self.intercept_ = self.intercept_ - self.learning_rate * loss_intercept

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
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.zeros(n_instances)
        bias = np.ones(n_instances)
        for i, sample in enumerate(X):
            product = self.coef_.dot(sample) + self.intercept_
            y[i] = product

        y = np.array(y > 0.0, dtype=int)

        return y

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
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        p = list()
        for sample in X:
            probability = sigmoid_function(self.coef_.dot(sample) + self.intercept_)
            p.append((1 - probability, probability))

        p = np.array(p)
        return p


if __name__ == "__main__":

    # Q5
    X_train, X_test, y_train, y_test = split_train_test()
    logistic_reg = LogisticRegressionClassifier()
    logistic_reg.fit(X_train, y_train)
    plot_boundary(
        "../images/LR",
        logistic_reg,
        X_test,
        y_test,
        mesh_step_size=0.1,
        title="Test logistic",
    )

    # Q6
    print("Q6) \n")

    n = 5
    scores = np.zeros(n)
    scores = {}

    for lr in [0.1, 0.5, 1, 2]:
        for n_iter in [5, 10, 50, 150]:
            scores[(lr, n_iter)] = np.zeros(n)
            for i in range(n):
                X_train, X_test, y_train, y_test = split_train_test(seed=i)
                logistic_reg = LogisticRegressionClassifier(
                    learning_rate=lr, n_iter=n_iter
                )
                logistic_reg.fit(X_train, y_train)
                scores[(lr, n_iter)][i] = logistic_reg.score(X_test, y_test)

    for (lr, n_iter), score in scores.items():
        print(
            "\tLearning rate: {}"
            "\n\tNumber of iteration: {}"
            "\n\tMean score: {} (+- {})".format(
                lr, n_iter, score.mean().round(4), score.std().round(4)
            )
        )
        print("-" * 50)

    # Q7
    print("\n Q7) \n")

    lr = 0.1

    for n_iter in [1, 5, 20, 100]:
        logistic_reg = LogisticRegressionClassifier(learning_rate=lr, n_iter=n_iter)

        logistic_reg.fit(X_train, y_train)
        plot_boundary(
            "../images/logistic_reg_{}_{}".format(n_iter, lr),
            logistic_reg,
            X_test,
            y_test,
        )

        print(
            "Accuracy for n_iter = {} and learning_rate = {} : {}".format(
                n_iter, lr, logistic_reg.score(X_test, y_test)
            ))


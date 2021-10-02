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

    Args:
        train_size (int, optional): Size of training set. Defaults to 1000.
        samples (int, optional): Samples' size. Defaults to 3000.
        seed (int): The random state

    Returns:
        X_train: array of shape [n_points, 2]
        X_test: array of shape [n_points, 2]
        y_train: array of shape [n_points]
        y_test: array of shape [n_points]
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
                sum_ += (sigmoid_function(self.theta_.dot(xi)) - yi) * xi

            loss = 1 / n_instances * sum_  # compute the loss
            self.theta_ = self.theta_ - self.learning_rate * loss  # update parameters
            # print(self.theta_)

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
        n_instances = X.shape[0]
        y = np.zeros(n_instances)
        bias = np.ones(n_instances)
        X_ = np.c_[bias, X]
        for i, sample in enumerate(X_):
            product = self.theta_.dot(sample)
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
        p = list()
        n_instances = X.shape[0]
        bias = np.ones(n_instances)
        X_ = np.c_[bias, X]
        for sample in X_:
            probability = sigmoid_function(self.theta_.dot(sample))
            p.append((1 - probability, probability))
        return np.array(p)
        pass


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_train_test()
    logistic_reg = LogisticRegressionClassifier()
    logistic_reg.fit(X_train, y_train)
    z = np.array([1.2, 1.3])
    plot_boundary(
        "../images/LR",
        logistic_reg,
        X_test,
        y_test,
        mesh_step_size=0.1,
        title="Test logistic",
    )

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
                lr, n_iter, score.mean().round(3), score.std().round(3)
            )
        )
        print("-" * 50)

    for n_iter in [5, 10, 50, 150]:
        print(n_iter)
        for lr in [0.1, 0.5, 1, 2]:
            print(
                "& {} (\pm{})".format(
                    (scores[(lr, n_iter)].mean()*100).round(2),
                    (scores[(lr, n_iter)].std()*100).round(2),
                )
            )

        print("\\\\")

    print("\hline")

    print(lr, n_iter)

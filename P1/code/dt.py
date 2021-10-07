"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1)

# Put your funtions here
# ...
import os

PROJECT_ROOT_DIR = ".."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")


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


def create_and_fit_models(
    min_samples_splits: list, seed: int, X: np.array, y: np.array
) -> list:
    """Create, fit and returns size(min_samples_splits) DecisionTreeClassifier models

    Args:
        min_samples_splits (list): array containing the min_samples_split parameter for each mode
        seed (int): random seed
        X (np.array): The inputs samples
        y (np.array): The output samples

    Returns:
        tree_clfs (list): List of tree classifer model
    """

    tree_clfs = []
    for min_samples_split in min_samples_splits:
        tree_clf = DecisionTreeClassifier(
            min_samples_split=min_samples_split, random_state=seed
        )
        tree_clf.fit(X, y)
        tree_clfs.append(tree_clf)

    return tree_clfs


def make_plots(
    tree_clf: DecisionTreeClassifier, X: np.array, y: np.array, plot_tree: bool = False,
):
    """Create plot in the "../images" directory

    Args:
        tree_clf (DecisionTreeClassifier): the model used for the plots
        X (np.array): The inputs samples
        y (np.array): The output samples
        plot_tree (bool): Plot the decision tree WARNING (need to install "graphviz")
    """

    samples_split = tree_clf.get_params()["min_samples_split"]
    plot_boundary(
        os.path.join(IMAGES_PATH, "dt_{}_samples_split".format(samples_split)),
        tree_clf,
        X,
        y,
    )

    if plot_tree:
        try:
            from graphviz import Source
            from sklearn.tree import export_graphviz

            export_graphviz(tree_clf, out_file="out.dot", rounded=True, filled=True)

            Source.from_file("out.dot")
            os.system(
                "dot -Tpng out.dot -o"
                + os.path.join(
                    IMAGES_PATH, "tree_{}_samples_split.png".format(samples_split)
                )
            )
            os.system("rm out.dot")
        except:
            print('Please install "graphviz" or set `plot_tree` to False')


if __name__ == "__main__":

    # Q1
    X_train, X_test, y_train, y_test = split_train_test()
    min_samples_splits = [2, 8, 32, 64, 128, 500]

    tree_clfs = create_and_fit_models(min_samples_splits, 42, X_train, y_train)
    for tree_clf in tree_clfs:
        make_plots(tree_clf, X_test, y_test, True)

    # Q3
    n = 5

    statistics = np.zeros((n, len(tree_clfs)))

    for i in range(n):
        X_train, X_test, y_train, y_test = split_train_test(seed=i)
        tree_clfs = create_and_fit_models(min_samples_splits, i, X_train, y_train)

        for j, tree_clf in enumerate(tree_clfs):
            statistics[i][j] = tree_clf.score(X_test, y_test)

    print("min_samples_splits: {}".format(min_samples_splits))
    print("mean accuracy associated: {}".format(np.mean(statistics, 0) * 100))
    print("and its standard deviation: {}".format(np.std(statistics, 0) * 100))
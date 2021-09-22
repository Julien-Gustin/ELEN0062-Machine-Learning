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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1)

# Put your funtions here
# ...
import os
from graphviz import Source
from sklearn.tree import export_graphviz

PROJECT_ROOT_DIR = ".."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")

def split_train_test(train_size=1000, samples=3000):
    """Return the train and test set

    Args:
        train_size (int, optional): Size of training set. Defaults to 1000.
        samples (int, optional): Samples' size. Defaults to 3000.

    Returns:
        X_train: array of shape [n_points, 2]
        X_test: array of shape [n_points, 2]
        y_train: array of shape [n_points]
        y_test: array of shape [n_points]
    """
    X_full, y_full = make_unbalanced_dataset(3000, random_state=42)

    X_train, X_test = X_full[:1000], X_full[1000:] 
    y_train, y_test = y_full[:1000], y_full[1000:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Put your code here

    X_train, X_test, y_train, y_test = split_train_test()

    for samples_split in [2, 8, 32, 64, 128, 500]:
        tree_clf = DecisionTreeClassifier(min_samples_split=samples_split, random_state=42)
        tree_clf.fit(X_train, y_train)
        plot_boundary(os.path.join(IMAGES_PATH, "dt_{}_samples_split".format(samples_split)), tree_clf, X_train, y_train)


        export_graphviz(
                tree_clf,
                out_file="out.dot",
                rounded=True,
                filled=True
            )

        Source.from_file("out.dot")
        os.system("dot -Tpng out.dot -o" + os.path.join(IMAGES_PATH, "tree_{}_samples_split.png".format(samples_split)))
        
    os.system("rm out.dot")
    pass

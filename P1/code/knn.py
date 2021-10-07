"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.core.numeric import cross
from sklearn import neighbors
import sklearn

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score


# (Question 2)

# Put your functions here
# ...
def split_train_test(X : list, y: list, training: int):
    X_train, y_train = X[:training], y[:training]
    X_test, y_test = X[training:], y[training:]
    return X_train, y_train, X_test, y_test

def print_results(n_neighbors_values : list, scores: list, scores_cross_val : list):
    for i in range (len(scores)):
        print("{} neighbor(s): Normal score ({}) Cross-Validation mean score ({})".format(n_neighbors_values[i], scores[i], scores_cross_val[i]))

def add_to_accuracy_fig(mean_test_accuracies: list):
    training_set_size = len(mean_test_accuracies)
    plt.plot([n for n in range (1, training_set_size+1)], mean_test_accuracies, label = "Training set size {}".format(training_set_size))

def save_best_neighbor_fig(filename : str, xlabel : str, ylabel : str, title: str, best_neighbors : list, training_set_sizes: list):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(training_set_sizes, best_neighbors)
    plt.savefig(filename)



if __name__ == "__main__":

    X, y = make_unbalanced_dataset(3000, random_state=42)
    X_train, y_train, X_test, y_test = split_train_test(X, y, 1000)
    n_neighbors_values = [1, 5, 50, 100, 500]
    
    #Q1
    scores = list()
    for n_neighbors in n_neighbors_values:
        #Create classifier
        neighbor_classifier = KNeighborsClassifier(n_neighbors)
        neighbor_classifier.fit(X_train, y_train)
        #Generate graph with boundaries
        file_name = "knn_" + str(n_neighbors) + "_neighbors"
        plot_boundary("../images/" + file_name, neighbor_classifier, X_test, y_test, 0.1, "Test KNN")
        #Memorize the score
        scores.append(neighbor_classifier.score(X_test, y_test))

    #Q2
    scores_cross_val = list()
    for n_neighbors in n_neighbors_values:
        #Create classifier 
        neighbor_classifier = KNeighborsClassifier(n_neighbors)
        #Compute score of each fold
        score_per_fold = cross_val_score(neighbor_classifier, X, y, cv=10)
        #Memorize the score
        scores_cross_val.append(np.mean(score_per_fold))

    print_results(n_neighbors_values, scores, scores_cross_val)

    #Q3
    test_set_size = 500
    training_set_sizes = [50, 150, 250, 350, 500]
    best_neighbors = list()
    seed = 1

    for training_set_size in training_set_sizes:
        mean_test_accuracies = np.zeros(training_set_size)
        for n in range(10):
            seed += 1
            X, y = make_unbalanced_dataset(training_set_size+test_set_size, random_state=(seed))
            X_train, y_train, X_test, y_test = split_train_test(X, y, training_set_size)
            
            for i in range(training_set_size):
                neighbor_classifier = KNeighborsClassifier(i+1)
                neighbor_classifier.fit(X_train, y_train)
                mean_test_accuracies[i] += neighbor_classifier.score(X_test, y_test)

        mean_test_accuracies = [acc / 10 for acc in mean_test_accuracies]
        best_neighbors.append(argmax(mean_test_accuracies) + 1)
        add_to_accuracy_fig(mean_test_accuracies)
    print(best_neighbors)

    #save plots
    plt.xlabel("Number of neighbors")
    plt.ylabel("Mean test accuracy")
    plt.title("Evolution of the mean test accuracies with different training set sizes")
    plt.legend()
    plt.savefig("../images/knn_mean_test_accuracies")
    plt.close()

    save_best_neighbor_fig(
        "../images/knn_best_neighbor_with_respect_to_training_set_size", 
        "Training size",
        "Optimal number of neighbors",
        "Optimal number of neighbors with respect to the training set size",
        best_neighbors,
        training_set_sizes)
    plt.close()
    pass


#Q1:
# Boundaries become smoother and smoother as k grows. With k = 1, each observation has a big impact on the model, meaning that the model doesn't generalise very well
# When taking higher values of k, the boundary decision separates the surface in two parts: a big blue surface and a big orange surface
# Therefore, when k grows, the model seems to say that the observations on the bottom left are classified as "Negative", 
# while the observations on the upper right are classified as "Positive".
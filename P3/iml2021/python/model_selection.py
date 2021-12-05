from sklearn.model_selection import LeaveOneGroupOut
import itertools
import numpy as np
from sklearn.base import clone

def crossValidationOneOut(X, y, estimator, groups):
    """Cross validation leave one subject out

    Returns list of scores for each fold out
    """
    scores = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(X, y, groups):
        print(np.unique(y[train_index]), np.unique(y[test_index]))
        X_train_fold, y_train_fold = X[train_index], y[train_index]
        X_test_fold, y_test_fold = X[test_index], y[test_index]
        estimator.fit(X_train_fold, y_train_fold)
        scores.append(estimator.score(X_test_fold, y_test_fold))
        print(scores[-1])
    return scores

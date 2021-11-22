from sklearn.model_selection import LeaveOneGroupOut
import itertools
import numpy as np
from sklearn.base import clone

def crossValidationOneOut(X, y, estimator, groups):
    scores = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(X, y, groups):
        X_train_fold, y_train_fold = X[train_index], y[train_index]
        X_test_fold, y_test_fold = X[test_index], y[test_index]
        estimator.fit(X_train_fold, y_train_fold)
        scores.append(estimator.score(X_test_fold, y_test_fold))

    return scores

class GridSearchGroup():
    def __init__(self, estimator, param_grid) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self._best_score = -np.inf
        self._best_estimator = None
        self._cv_results = {"mean_test_score": [], "params": []}

    def fit(self, X, y, groups):
        for set_of_parameters in self.param_grid:
            keys = set_of_parameters.keys()
            for params in itertools.product(*set_of_parameters.values()):
                input_parameters = dict(zip(keys ,params))
                estimator = clone(self.estimator)
                estimator.set_params(**input_parameters)
                score = np.mean(crossValidationOneOut(X, y, estimator, groups))
                self._cv_results["mean_test_score"] = score
                self._cv_results["params"] = params

                if score > self._best_score:
                    print(score, input_parameters)
                    self._best_score = score
                    self._best_estimator = estimator

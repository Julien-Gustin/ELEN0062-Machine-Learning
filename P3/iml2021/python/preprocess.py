from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
)
from scipy import signal
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from tsfresh.feature_extraction import extract_features

class TimeSerieMaker(BaseEstimator, TransformerMixin):
    """Transform data (np.array) into time series (in order to be used by sktime)

    Returns:
        Panda time series
    """
    def __init__(self):
        self.std_list = []
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        time_series = []

        for _, xi in enumerate(X_):
            time_series.append(np.split(xi, 31))

        time_series = np.array(time_series) # [sample][sensor]
        nested_time_series = from_3d_numpy_to_nested(time_series)
        return nested_time_series

class MedianFilter(BaseEstimator, TransformerMixin):
    """Given a Panda time series compute the median filter of triplet of column

    Returns:
        Panda time series
    """
    def __init__(self, columns, kernel=3):
        self.columns = columns
        self.kernel = kernel
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        for g in self.columns:
            X_["var_{}".format(g[0])] = X_["var_{}".format(g[0])].apply(signal.medfilt, self.kernel)
            X_["var_{}".format(g[1])] = X_["var_{}".format(g[1])].apply(signal.medfilt, self.kernel)
            X_["var_{}".format(g[2])] = X_["var_{}".format(g[2])].apply(signal.medfilt, self.kernel)

        return X_

class NestedTo2dArray(BaseEstimator, TransformerMixin):
    """Transform panda time series to 2d np array

    Returns:
        np.array
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = from_nested_to_2d_array(X, return_numpy=True)
        return X


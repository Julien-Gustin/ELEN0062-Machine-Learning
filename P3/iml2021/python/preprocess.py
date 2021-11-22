from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
)

class CleanNanByMedian(BaseEstimator, TransformerMixin):
    """Clean the data by replacing Nan (-999999.99) by the median of the column
    """
    def __init__(self):
        self.median = 0
        pass

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.median = X_df.median()
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_df = pd.DataFrame(X_)
        X_clean = np.array(X_df.replace(-999999.99, np.nan).fillna(self.median))
        return X_clean

class CleanNanByMedianV2(BaseEstimator, TransformerMixin):
    """Clean the data by replacing Nan (-999999.99) by the median of the column OF THE SAME ACTIVITY
    """
    def __init__(self):
        self.medians = [] # Activities go from 1 to 14, (need to do -1 for the list)
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y):
        X_ = X.copy()
        for yi in np.unique(y): # for each activity
            Xyi = X_[y == yi] 
            Xyi_df = pd.DataFrame(Xyi)
            self.medians.append(Xyi_df.median()) # compute the median for each column (of a same activity)

        return self

    def transform(self, X, y):
        X_ = X.copy()
        X_df = pd.DataFrame(X_).replace(-999999.99, np.nan)
        for yi in np.unique(y): # for each activity
            yi = int(yi)
            rows_yi = np.where(y == yi)[0]
            X_df.loc[rows_yi] = X_df.loc[rows_yi].fillna(value=self.medians[yi-1])
       
        X_clean = np.array(X_df)
        return X_clean

class TimeSerieMaker(BaseEstimator, TransformerMixin):
    """Transform data into time series (in order to be used by sktime)

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

class SimpleSlidingWindowFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, time_serie_length=512):
        if (time_serie_length % window_size) != 0:
            raise Exception("window_size should be a diviser of time_serie_length")
        self.window_size = window_size 
        self.time_serie_length = time_serie_length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        _, columns = X_.shape

        X_output = np.empty((len(X_)))
        for window_nb in range(columns//self.window_size):
            stds = X_[:,self.window_size*window_nb:(1+window_nb)*self.window_size].std(axis=1)
            means = X_[:,self.window_size*window_nb:(1+window_nb)*self.window_size].mean(axis=1)
            slopes = (X_[:,window_nb] - X_[:,(1+window_nb)*self.window_size -1])/self.window_size

            X_output = np.c_[(X_output, stds, means, slopes)]

        X_output = X_output.T[1:].T # remove `empty` elements

        return X_output
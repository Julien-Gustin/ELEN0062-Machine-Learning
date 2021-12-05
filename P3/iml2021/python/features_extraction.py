from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features


class AdvancedSlidingWindowFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=512, overlap=0.5, std=False, mean=False, median=False, mini=False, maxi=False, slope=False):
        self.window_size = window_size
        self.overlap = overlap 
        self.std = std
        self.mean = mean
        self.median = median
        self.mini = mini
        self.maxi = maxi
        self.slope = slope
        
    def fit(self, X, y=None):
        return self

    def sliding_windows(self, ts):
        heigh, width = ts.shape
        start = 0
        end = self.window_size
        step = int(self.window_size * (1 - self.overlap))
        X_output = np.empty(heigh)
        for _ in range(width//step):
            if self.std:
                X_output = np.c_[(X_output, ts[:, start:end].std(axis=1))]

            if self.mean:
                X_output = np.c_[(X_output, ts[:, start:end].mean(axis=1))]

            if self.median:
                X_output = np.c_[(X_output, np.quantile(ts[:, start:end], 0.5, axis=1))]

            if self.mini:
                X_output = np.c_[(X_output, np.min(ts[:, start:end], axis=1))]

            if self.maxi:
                X_output = np.c_[(X_output, np.max(ts[:, start:end], axis=1))]

            if self.slope:
                X_output = np.c_[(X_output, (ts[:, start] - ts[:, end-1])/float(self.window_size))]

            start += step
            end = min(width, start + self.window_size)

        X_output = X_output.T[1:].T # remove `empty` elements
        return X_output

    def transform(self, X, y=None):
        X_ = X.copy()

        features = []
        for split in np.split(X_, 31, axis=1):
            features.append(self.sliding_windows(split))

        X_f = np.concatenate(features, axis=1)

        return X_f


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """Extract features with tsresh"""
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        splitted_x = np.split(X, 31, axis=1)
        for i in range(len(splitted_x)):
            df = pd.DataFrame(splitted_x[i])
            df["id"] = df.index
            df = df.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)
            X = extract_features(df, column_id="id", column_sort="time", default_fc_parameters=self.parameters)
            features.append(np.array(X))

        X_out = np.concatenate(features, axis=1)
        return X_out
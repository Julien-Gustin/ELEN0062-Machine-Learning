import numpy as np
from sklearn.base import clone
import utils

class Protocol:
    def __init__(self, X, y, n_LS, models) -> None:
        self.X = X
        self.y = y
        self.n_LS = n_LS
        self.noise = None
        self.variance = None
        self.bias_2 = None
        self.models_kind = models

    def split_data_set(self):
        """Split data set in 'n_LS' dataset

        Returns:
            [Tuple]: [X_split, y_split]
        """
        X_split = np.split(self.X, self.n_LS)
        y_split = np.split(self.y, self.n_LS)

        return X_split, y_split

    def get_noise(self, x0, scale=0.5): # step (iii)
        """ Get an approximation of the noise given a (sorted) array of x0

        Returns:
            [np.array]: noise associated to each x0
        """
        if self.noise is not None:
            return self.noise

        y0 = []
        for i in range(1000):
            y0.append(utils.f(x0, scale=scale, seed=i))

        y0 = np.array(y0)
        self.noise = np.var(y0, axis=0, ddof=1)

        return self.noise

    def get_mean(self, x0): # step (ii)
        """ Get the mean of y given an array of (sorted) x0

        Returns:
            [np.array]: mean of y for each xi
        """

        y0 = []
        for i in range(1000):
            y0.append(utils.f(x0, seed=i))

        y0 = np.array(y0)

        return np.mean(y0, axis=0)

    def get_variance_bias(self, x0):
        """ Get the variance and squared bias of each x0 for all `k` model in `models`

        Returns:
            [tuple]: [[variance_m0, ... variance_mk], [bias^2_m0, ..., [bias^2_mk]]
        """
        if self.variance is not None and self.bias_2 is not None:
            return self.variance, self.bias_2

        self.variance = []
        self.bias_2 = []

        X_split, y_split = self.split_data_set() # step (iv)

        y0_mean = self.get_mean(x0)

        for model in self.models_kind:
            predictions = []

            for sub_x, sub_y in zip(X_split, y_split): 
                model_i = clone(model)
                sub_x = sub_x.reshape(-1, 1)
                sub_y = sub_y.reshape(-1, 1)
                model_i.fit(sub_x, sub_y) # step (v)

                y_pred = model_i.predict(x0.reshape(-1, 1)) # step (vi)
                predictions.append(y_pred)

            self.variance.append(np.var(predictions, axis=0, ddof=1).flatten())
            self.bias_2.append((np.mean(predictions, axis=0).flatten() - y0_mean)**2)

        return self.variance, self.bias_2

    def get_expected_error(self, x0):
        """ Get the expected error for each `unique` xi for all `k` model in `models`

        Returns:
            [np.array]: [expected_error_m0, ... expected_error_mk]
        """
        if self.variance is None or self.bias_2 is None:
            self.get_variance_bias(x0)

        if self.noise is None:
            self.get_noise(x0)

        return self.noise + self.variance + self.bias_2

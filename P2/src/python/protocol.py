import numpy as np
from sklearn.base import clone

class Protocol:
    def __init__(self, X, y, n_LS, models) -> None:
        self.X = X
        self.y = y
        self.n_LS = n_LS
        self.noise = None
        self.variance = None
        self.bias_2 = None
        self.models_kind = models

    def get_unique(self):
        """Get an array containing unique xi in self.X

        Returns:
            [np.array]: unique xi
        """
        return np.unique(self.X)

    def split_data_set(self):
        """Split data set in 'n_LS' dataset

        Returns:
            [Tuple]: [X_split, y_split]
        """
        X_split = np.split(self.X, self.n_LS)
        y_split = np.split(self.y, self.n_LS)

        return X_split, y_split

    def get_noise(self): # step (iii)
        """ Get an approximation of the noise for each `unique` xi

        Returns:
            [np.array]: noise associated to each `unique` xi
        """
        if self.noise is not None:
            return self.noise

        X_unique = self.get_unique()
        noise = np.zeros(len(X_unique))
        for i, xi in enumerate(X_unique):
            indexes = self.X == xi
            y_xi = self.y[indexes]
            noise[i] = np.var(y_xi)

        self.noise = noise
        return noise

    def get_mean(self): # step (ii)
        """ Get the mean of y given `unique` xi

        Returns:
            [np.array]: mean of y for each xi
        """
        X_unique = self.get_unique()
        y = np.zeros(len(X_unique))
        for i, xi in enumerate(X_unique):
            indexes = self.X == xi
            y[i] = self.y[indexes].mean()

        return y

    def get_variance_bias(self):
        """ Get the variance and squared bias of each `unique` xi for all `k` model in `models`

        Returns:
            [tuple]: [[variance_m0, ... variance_mk], [bias^2_m0, ..., [bias^2_mk]]
        """
        if self.variance is not None and self.bias_2 is not None:
            return self.variance, self.bias_2

        self.variance = []
        self.bias_2 = []

        X_split, y_split = self.split_data_set() # step (iv)
        X_unique = self.get_unique()
        y_mean = self.get_mean()

        for model in self.models_kind:
            predictions = []

            for sub_x, sub_y in zip(X_split, y_split): 
                model_i = clone(model)
                sub_x = sub_x.reshape(-1, 1)
                sub_y = sub_y.reshape(-1, 1)
                model_i.fit(sub_x, sub_y) # step (v)

                y_pred = model_i.predict(X_unique.reshape(-1, 1)) # step (vi)
                predictions.append(y_pred)

            self.variance.append(np.var(predictions, axis=0).flatten())
            self.bias_2.append((np.mean(predictions, axis=0).flatten() - y_mean)**2)

        return self.variance, self.bias_2

    def get_expected_error(self):
        """ Get the expected error for each `unique` xi for all `k` model in `models`

        Returns:
            [np.array]: [expected_error_m0, ... expected_error_mk]
        """
        if self.variance is None or self.bias_2 is None:
            self.get_variance_bias()

        if self.noise is None:
            self.get_noise()

        return self.noise + self.variance + self.bias_2

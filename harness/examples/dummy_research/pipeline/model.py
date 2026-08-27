import numpy as np


class Model:
    name = "persistence"

    def fit(self, X, y, sample_weight=None):
        del X, y, sample_weight
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]

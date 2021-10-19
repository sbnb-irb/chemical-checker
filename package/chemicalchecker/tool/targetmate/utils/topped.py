import numpy as np
from scipy.stats import mode
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


class ToppedClassifier(object):

    def __init__(self, clf, max_samples=10000, max_ensemble_size=30,
                 chance=0.95):
        """Top the training of a classifier to a limited number of samples.

        Args:
            clf: A classifier instance.
            max_samples (int): Maximum number of samples (default=10000).
            max_ensemble_size (int): Maximum size of the ensemble,
                i.e. maximum number or runs (default=30).
            chance (float): Chance of seing the sample (default = 0.95).
        """
        self.max_samples = max_samples
        self.max_ensemble_size = max_ensemble_size
        self.chance = chance
        self.clf = clf
        self.clfs = []

    @staticmethod
    def get_resamp(s, N, chance):
        p = 1 - float(s) / N
        return np.log(1 - chance) / np.log(p)

    def calc_ensemble_size(self, X, y):
        N = X.shape[0]
        if N <= self.max_samples:
            self.samples = N
            self.ensemble_size = 1
        else:
            self.samples = self.max_samples
            self.ensemble_size = int(
                np.ceil(self.get_resamp(self.samples, N, self.chance)))

    def fit(self, X, y=None):
        self.calc_ensemble_size(X, y)
        if self.ensemble_size == 1:
            clf = clone(self.clf)
            if y is None:
                clf.fit(X)
            else:
                clf.fit(X, y)
            self.clfs += [clf]
        else:
            if y is None:
                splits = ShuffleSplit(
                    n_splits=self.ensemble_size, train_size=self.samples)
            else:
                splits = StratifiedShuffleSplit(
                    n_splits=self.ensemble_size, train_size=self.samples)
            for train_idxs, _ in splits.split(X, y):
                clf = clone(self.clf)
                X_train = X[train_idxs]
                if y is None:
                    clf.fit(X_train)
                else:
                    y_train = y[train_idxs]
                    clf.fit(X_train, y_train)
                self.clfs += [clf]

    def predict(self, X):
        y = np.zeros((X.shape[0], len(self.clfs)))
        for j, clf in enumerate(self.clfs):
            y[:, j] = clf.predict(X)
        y = mode(y, axis=1)[0].ravel()
        return y

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], len(self.clfs)))
        for j, clf in enumerate(self.clfs):
            y[:, j] = clf.predict_proba(X)[:, 1]
        y = np.mean(y, axis=1)
        return y

    def explain(self, X):
        pass


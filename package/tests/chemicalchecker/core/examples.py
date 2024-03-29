"""Example data.

This class fetches examples to be used in tutorials and notebooks.
"""
import os
import csv
import pickle
import numpy as np
import pandas as pd

from chemicalchecker.util import logged


class BaseExample(object):
    """BaseExample class."""

    def __init__(self, dup_keys_prop, dup_features_prop,
                 nan_prop, inf_prop,
                 empty_keys_prop, empty_features_prop,
                 force):
        """Initialize a BaseExample instance."""
        self.path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "examples")
        self.dup_keys_prop = dup_keys_prop
        self.dup_features_prop = dup_features_prop
        self.nan_prop = nan_prop
        self.inf_prop = inf_prop
        self.empty_keys_prop = empty_keys_prop
        self.empty_features_prop = empty_features_prop
        self.force = force

    def check_type(self, key_type):
        if key_type not in ["inchikey", "smiles", "src"]:
            raise Exception(
                "key_type must be one of 'inchikey', 'smiles' and 'src'")

    def duplicate_keys_pairs(self, pairs):
        keys = list(set([p[0] for p in pairs]))
        num = int(len(keys) * self.dup_keys_prop)
        if num == 0:
            return pairs
        else:
            features = [p[1] for p in pairs]
            keys_ = np.random.choice(keys, num, replace=False)
            features_ = np.random.choice(features, num, replace=True)
            for k, f in zip(keys_, features_):
                pairs += [(k, f)]
            return pairs

    def duplicate_features_pairs(self, pairs):
        features = list(set([p[1] for p in pairs]))
        num = int(len(features) * self.dup_features_prop)
        if num == 0:
            return pairs
        else:
            keys = [p[0] for p in pairs]
            features_ = np.random.choice(features, num, replace=False)
            keys_ = np.random.choice(keys, num, replace=True)
            for k, f in zip(keys_, features_):
                pairs += [(k, f)]
            return pairs

    def duplicate_keys_matrix(self, X, keys):
        num = int(X.shape[0] * self.dup_keys_prop)
        if num == 0:
            return X, keys
        else:
            keys_ = np.random.choice(keys, num, replace=False)
            idxs_ = np.randam.choice(X.shape[0], num, replace=False)
            X_ = X[idxs_]
            X = np.vstack([X, X_])
            keys = np.array(list(keys) + list(keys_))
            return X, keys

    def duplicate_features_matrix(self, X, features):
        num = int(X.shape[1] * self.dup_features_prop)
        if num == 0:
            return X, features
        else:
            features_ = np.random.choice(features, num, replace=False)
            idxs_ = np.random.choice(X.shape[1], num, replace=False)
            X_ = X[:, idxs_]
            X = np.hstack([X, X_])
            features = np.array(list(features) + list(features_))
            return X, features

    def add_nans(self, X):
        if not self.force:
            if isinstance(X[0, 0], int):
                return X
        num = int(X.shape[0] * X.shape[1] * self.nan_prop)
        if num == 0:
            return X
        else:
            X = X.astype(float)
            idxs1 = np.random.choice(X.shape[0], num, replace=True)
            idxs2 = np.random.choice(X.shape[1], num, replace=True)
            for i1, i2 in zip(idxs1, idxs2):
                X[i1, i2] = np.nan
            return X

    def add_infs(self, X):
        if not self.force:
            if isinstance(X[0, 0], int):
                return X
        num = int(X.shape[0] * X.shape[1] * self.nan_prop)
        if num == 0:
            return X
        else:
            X = X.astype(float)
            idxs1 = np.random.choice(X.shape[0], num, replace=True)
            idxs2 = np.random.choice(X.shape[1], num, replace=True)
            dires = np.random.choice(2, num, replace=True)
            for i1, i2, dr in zip(idxs1, idxs2, dires):
                if dr == 0:
                    X[i1, i2] = np.inf
                else:
                    X[i1, i2] = -np.inf
            return X

    def empty_keys(self, X):
        num = int(X.shape[0] * self.empty_keys_prop)
        if num == 0:
            return X
        else:
            idxs = np.random.choice(X.shape[0], num, replace=False)
            X[idxs, :] = 0
            return X

    def empty_features(self, X):
        num = int(X.shape[1] * self.empty_features_prop)
        if num == 0:
            return X
        else:
            idxs = np.random.choice(X.shape[1], num, replace=False)
            X[:, idxs] = 0
            return X

    def returner_matrix(self, X, keys, features):
        X, keys = self.duplicate_keys_matrix(X, keys)
        X, features = self.duplicate_features_matrix(X, features)
        X = self.add_nans(X)
        X = self.add_infs(X)
        X = self.empty_keys(X)
        X = self.empty_features(X)
        return X, keys, features

    def returner_pairs(self, pairs):
        pairs = self.duplicate_keys_pairs(pairs)
        pairs = self.duplicate_features_pairs(pairs)
        return pairs


@logged
class Example(BaseExample):
    """Example class.

    The Example class contains some example data, accessible with the class
    methods.
    """

    def __init__(self, dup_keys_prop=0, dup_features_prop=0,
                 nan_prop=0, inf_prop=0,
                 empty_keys_prop=0, empty_features_prop=0,
                 force=False):
        """Initialize a Example instance."""
        BaseExample.__init__(self, dup_keys_prop, dup_features_prop,
                             nan_prop, inf_prop,
                             empty_keys_prop, empty_features_prop,
                             force)

    def toy_matrix(self, key_type='inchikey', fit_case=True):
        """A toy matrix, just to test the sign0.

        Args:
            key_type(str): One of 'inchikey', 'smiles', 'src' (source)
                (default='inchikey')
            fit_case(bool): Get a fit case or a predict/transform case
                (default=True)

        Returns:
            X, keys, features
        """
        self.check_type(key_type)
        self.__log.info("Getting a toy matrix dataset")
        with open(os.path.join(self.path, "toy_matrix.pkl"), "rb") as f:
            d = pickle.load(f)
        if fit_case:
            suf = 1
        else:
            suf = 2
        X = d["X%d" % suf]
        features = d["features%d" % suf]
        if key_type == "inchikey":
            keys = d["key_inks%d" % suf]
        if key_type == "smiles":
            keys = d["key_smis%d" % suf]
        if key_type == "src":
            keys = d["key_srcs%d" % suf]
        return self.returner_matrix(X, keys, features)

    def toy_pairs(self, key_type='inchikey', fit_case=True):
        """Toy pairs, just to test the sign0.

        Args:
            key_type(str): One of 'inchikey', 'smiles', 'src' (source)
                (default='inchikey')
            fit_case(bool): Get a fit case or a predict/transform case
                (default=True)

        Returns:
            pairs
        """
        self.check_type(key_type)
        self.__log.info("Getting a toy pairs dataset")
        with open(os.path.join(self.path, "toy_pairs.pkl"), "rb") as f:
            d = pickle.load(f)
        if fit_case:
            suf = 1
        else:
            suf = 2
        if key_type == "inchikey":
            pairs = d["pairs_inks%d" % suf]
        if key_type == "smiles":
            pairs = d["pairs_smis%d" % suf]
        if key_type == "src":
            pairs = d["pairs_srcs%d" % suf]
        return self.returner_pairs(pairs)

    def drug_targets(self):
        """Drug targets from PharmacoDB.

        Returns:
            pairs
        """
        self.__log.info("Getting drug targets (example of unweighted pairs)")
        pairs = []
        with open(os.path.join(self.path, "pharmacodb_targets.tsv"), "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                pairs += [(r[0], r[1])]
        return self.returner_pairs(pairs)

    def cell_sensitivity(self):
        """Cell sensitivity data from CTRP

            Returns:
                X, keys, features
        """
        self.__log.info(
            "Getting cell sensitivity data (example of continuous data)")
        df = pd.read_csv(os.path.join(
            self.path, "ctrp_auc.csv"), delimiter=",")
        keys = np.array(df[df.columns[0]]).astype(str)
        features = np.array(df.columns[1:]).astype(str)
        X = np.array(df[df.columns[1:]]).astype(float)
        return self.returner_matrix(X, keys, features)

    def fingerprints(self):
        """Morgan fingerprints from LINCS.

        Returns:
            X, keys, features
        """
        self.__log.info("Getting fingerprint data for LINCS molecules")
        with open(os.path.join(self.path, "fps.pkl"), "rb") as f:
            d = pickle.load(f)
            X = d["V"]
            keys = np.array(d["keys"])
            features = np.array(d["features"])
        return self.returner_matrix(X, keys, features)

import os
import numpy as np
import pickle
from chemicalchecker.util import logged
from ..utils import metrics


def load_validation(destination_dir):
    with open(destination_dir, "rb") as f:
        return pickle.load(f)


class PrecomputedSplitter:
    """Useful when train and test indices are pre-specified."""
    
    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(X = None, y = None):
        yield self.train_idx, test_idx


@logged
class Performances:
    """Calculate performances"""

    def __init__(self, is_classifier):
        self.is_classifier = is_classifier

    @staticmethod
    def truth_inv(yt):
        """Given an active (1) / inactive (0) dataset, assign 1 to inactives"""
        return np.abs(1 - yt)

    def classifier_performances(self, yt, yp):
        """Calculate standard prediction performance metrics.
        In addition, it calculates the corresponding weights.
        For the moment, AUPR and AUROC are used.
        Args:
            yt(array): Truth data (binary).
            yp(array): Prediction scores (probabilities).
        """
        shape = yp.shape
        if len(shape) == 1:
            self.__log.debug("Prediction has only one dimension, i.e. score for active")
            self.auroc  = metrics.roc_score(yt, yp)[0]
            self.aupr   = metrics.pr_score(yt, yp)[0]
            self.bedroc = metrics.bedroc_score(yt, yp)[0]
        else:
            assert shape[1] == 2, "Too many classes. For the moment, TargetMate only works with active/inactive (1/0)"
            assert set(yt) == set([0,1]), "Only 1/0 is accepted for the moment."
            if len(shape) == 2:
                self.__log.debug("Prediction has two dimensions (active/inactive). Not ensemble.")
                self.auroc  = np.zeros(2)
                self.aupr   = np.zeros(2)
                self.bedroc = np.zeros(2)
                self.__log.debug("Measuring performance for inactives")
                yt_ = self.truth_inv(yt)
                self.auroc[0]  = metrics.roc_score(yt_, yp[:,0])[0]
                self.aupr[0]   = metrics.pr_score(yt_, yp[:,0])[0]
                self.bedroc[0] = metrics.bedroc_score(yt_, yp[:,0])[0]
                self.__log.debug("Measuring performances for actives")
                yt_ = yt
                self.auroc[1]  = metrics.roc_score(yt_, yp[:,1])[0]
                self.aupr[1]   = metrics.pr_score(yt_, yp[:,1])[0]
                self.bedroc[1] = metrics.bedroc_score(yt_, yp[:,1])[0]
            if len(shape) == 3:
                self.__log.debug("Prediction has three dimensions (active/inactives and ensemble of datasets)")
                ens_size = shape[2]
                self.auroc  = np.zeros((2, ens_size))
                self.aupr   = np.zeros((2, ens_size))
                self.bedroc = np.zeros((2, ens_size))
                self.__log.debug("Measuring performance for inactives")
                yt_ = self.truth_inv(yt)
                for k in range(0, ens_size):
                    self.auroc[0, k]  = metrics.roc_score(yt_, yp[:,0,k])[0]
                    self.aupr[0, k]   = metrics.pr_score(yt_, yp[:,0,k])[0]
                    self.bedroc[0, k] = metrics.bedroc_score(yt_, yp[:,0,k])[0]
                self.__log.debug("Measuring performance for inactives")
                yt_ = yt
                for k in range(0, ens_size):
                    self.auroc[1, k]  = metrics.roc_score(yt_, yp[:,1,k])[0]
                    self.aupr[1, k]   = metrics.pr_score(yt_, yp[:,1,k])[0]
                    self.bedroc[1, k] = metrics.bedroc_score(yt_, yp[:,1,k])[0]

    def regressor_performances(self, yt, yp):
        """ """
        pass

    def compute(self, yt, yp):
        if self.is_classifier:
            self.classifier_performances(yt, yp)
        else:
            self.regressor_performances(yt, yp)
        return self

    def as_dict(self):
        if self.is_classifier:
            perfs_dict = {
                "auroc" : self.auroc,
                "aupr"  : self.aupr,
                "bedroc": self.bedroc
            }
        else:
            perfs_dict = {
                "x": "x"
            }
        return perfs_dict


@logged
class Validation:
    """Validation class."""

    def __init__(self, splitter=None, destination_dir=None):
        """Initialize validation class.

        Args:
            splitter(): If none specified, the corresponding TargetMate splitter is used (default=None).
            destination_dir(str): If non specified, the models path of the TargetMate instance is used (default=None).
        """
        self.splitter = splitter
        self.destination_dir = destination_dir

    @staticmethod
    def stack(ar_a, ar_b):
        if ar_a is None:
            return ar_b
        else:
            if len(ar_b.shape) == 1:
                return np.hstack((ar_a, ar_b))
            else:
                return np.vstack((ar_a, ar_b))

    def compute(self, tm, data, train_idx, test_idx):
        """Do the cross-validation"""
        # Initialize
        self.is_classifier = tm.is_classifier
        self.is_ensemble = tm.is_ensemble
        self.conformity = tm.conformity
        self.weights = tm.weights
        self.smiles = data.smiles
        # Setup
        self.__log.debug("Initializing all the results arrays")
        self.train_idx = None
        self.test_idx = None
        self.y_true_train = None
        self.y_true_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        if self.is_ensemble:
            self.y_pred_ens_train = None
            self.y_pred_ens_test = None
        # Signaturize
        self.__log.info("Signaturizing all data")
        tm.signaturize(data.smiles, is_tmp=True)
        # Splits
        self.__log.debug("Computing validation")
        if train_idx is not None and test_idx is not None:
            kf = PrecomputedSplitter(train_idx, test_idx)
        else:
            if not self.splitter:
                kf = tm.kfolder()
            else:
                kf = self.splitter
        i = 0
        for train_idx, test_idx in kf.split(X=np.zeros(len(data.activity)), y=data.activity):
            self.__log.info("Fold %0d" % (i+1))
            data_train = data[train_idx]
            data_test  = data[test_idx]
            self.__log.debug("Fit with train set")
            tm.fit(data, idxs=train_idx, is_tmp=True)
            self.__log.debug("Predict with train set")
            pred_train = tm.predict(data, idxs=train_idx, is_tmp=True)
            self.__log.debug("Predict with test set")
            pred_test = tm.predict(data, idxs=test_idx, is_tmp=True)
            self.__log.debug("Appending")
            self.train_idx = self.stack(self.train_idx, train_idx)
            self.test_idx = self.stack(self.test_idx, test_idx)
            self.y_true_train = self.stack(self.y_true_train, data_train.activity)
            self.y_true_test = self.stack(self.y_true_test, data_test.activity)
            self.y_pred_train = self.stack(self.y_pred_train, pred_train.y_pred)
            self.y_pred_test = self.stack(self.y_pred_test, pred_test.y_pred)
            if self.is_ensemble:
                self.y_pred_ens_train = self.stack(self.y_pred_ens_train, pred_train.y_pred_ens)
                self.y_pred_ens_test = self.stack(self.y_pred_ens_test, pred_test.y_pred_ens)
            i += 1

    def score(self):
        self.perfs_train = Performances(self.is_classifier).compute(self.y_true_train, self.y_pred_train)
        self.perfs_test  = Performances(self.is_classifier).compute(self.y_true_test,  self.y_pred_test )
        if self.is_ensemble:
            self.perfs_ens_train = Performances(self.is_classifier).compute(self.y_true_train, self.y_pred_ens_train)
            self.perfs_ens_test  = Performances(self.is_classifier).compute(self.y_true_test,  self.y_pred_ens_test )

    def as_dict(self):
        valid = {
            "is_classifier": self.is_classifier,
            "is_ensemble": self.is_ensemble,
            "conformity": self.conformity,
            "datasets": self.datasets,
            "weights": self.weights,
            "smiles": self.smiles,
            "destination_dir": self.destination_dir,
            "train": {
                    "y_true": self.y_true_train,
                    "y_pred": self.y_pred_train,
                    "perfs" : self.perfs_train.as_dict()
                },
            "test": {
                    "y_true": self.y_true_test,
                    "y_pred": self.y_pred_test,
                    "perfs" : self.perfs_test.as_dict()
            }
        }
        if self.is_ensemble:
            valid["ens_train"] = {
                "y_true": self.y_true_train,
                "y_pred": self.y_pred_ens_train,
                "perfs" : self.perfs_ens_train.as_dict()
            }
            valid["ens_test"]  = {
                "y_true": self.y_true_test,
                "y_pred": self.y_pred_ens_test,
                "perfs" : self.perfs_ens_test.as_dict()
            }
        return valid
    
    def save(self, d=None):
        """Save. If d is specified, a dictionary is saved."""
        self.__log.debug("Saving validation")
        with open(self.destination_dir, "wb") as f:
            if d is None:
                pickle.dump(self, f)
            else:
                pickle.dump(d, f)

    def validate(self, tm, data, train_idx=None, test_idx=None, as_dict=True, save=True):
        """Validate a TargetMate classifier using train-test splits.

        Args:
            tm(TargetMate model): The TargetMate model to be evaluated.
            data(InputData): Data object.
            train_idx(array): Precomputed indices for the train set (default=None). 
            test_idx(array): Precomputed indices for the test set (default=None).
            as_dict(bool): Return as dictionary, for portability; if False, the validation is done inplace (default=True).
            save(bool): Save (default=True)
        """
        if self.destination_dir is None:
            self.destination_dir = os.path.join(tm.models_path, "validation.pkl")
        else:
            self.destination_dir = self.destination_dir
        self.datasets = tm.datasets
        self.compute(tm, data, train_idx, test_idx)
        self.score()
        if as_dict:
            d = self.as_dict()
        else:
            d = None
        if save:
            self.save(d = d)
        return d

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

    def classifier_performances(self, yt, yp):
        """Calculate standard prediction performance metrics.
        In addition, it calculates the corresponding weights.
        For the moment, AUPR and AUROC are used.
        Args:
            yt(list): Truth data (binary).
            yp(list): Prediction scores (probabilities).
        """
        perfs = {}
        yt = list(yt)
        yp = list(yp)
        self.auroc = metrics.roc_score(yt, yp)
        self.aupr = metrics.pr_score(yt, yp)
        self.bedroc = metrics.bedroc_score(yt, yp)
  
    def regressor_performances(self, yt, yp):
        """ """
        pass

    def compute(self, yt, yp):
        if self.is_classifier:
            self.classifier_performances(yt, yp)
        else:
            self.regressor_performances(yt, yp)


@logged
class Validation:
    """Validation class."""

    def __init__(self, splitter=None, destination_dir=None):
        """Initialize validation class.

        Args:
            splitter(): If none specified, the corresponding TargetMate splitter is used (default=None).
            destination_dir(str): If non specified, the models path of the TargetMate instance is used (default=None).
        """
        self.destination_dir = destination_dir
        self.splitter   = splitter

    @staticmethod
    def stack(ar_a, ar_b):
        if ar_a is None:
            return ar_b
        else:
            return np.vstack(ar_a, ar_b)

    def compute(self, tm, data, train_idx, test_idx):
        """Do the cross-validation"""
        # Initialize
        self.is_ensemble = tm.is_ensemble
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
            tm.fit(data_train, is_tmp=True)
            self.__log.debug("Predict with train set")
            pred_train = tm.predict(data_train)
            self.__log.debug("Predict with test set")
            pred_test = tm.predict(data_test)
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
        self.perfs_test = Performances(self.is_classifier).compute(self.y_true_test, self.y_true_test)
        if self.is_ensemble:
            self.perfs_ens_train = Performances(self.is_classifier).compute(self.y_true_train, self.y_pred_ens_train)
            self.perfs_ens_test = Performances(self.is_classifier).compute(self.y_true_test, self.y_pred_ens_test)

    def save(self):
        self.__log.debug("Saving validation")
        with open(self.destination_dir, "wb") as f:
            pickle.dump(self, f)

    def validate(self, tm, data, train_idx=None, test_idx=None, save=True):
        """Validate a TargetMate classifier using train-test splits.

        Args:
            tm(TargetMate model): The TargetMate model to be evaluated.
            data(InputData): Data object.
            train_idx(array): Precomputed indices for the train set (default=None). 
            test_idx(array): Precomputed indices for the test set (default=None).
        """
        if self.destination_dir is None:
            destination_dir = os.path.join(tm.models_path, "validation.pkl")
        else:
            destination_dir = self.destination_dir
        self.datasets = tm.datasets
        self.compute(tm, data, train_idx, test_idx)
        self.score()
        if save:
            self.save()
"""Validate a TargetMate model (regression or classifier)"""
import os
import numpy as np
import h5py
import pickle
from chemicalchecker.util import logged
from sklearn import model_selection
from ..utils import metrics
from ..utils import HPCUtils


def load_validation(destination_dir):
    with open(destination_dir, "rb") as f:
        return pickle.load(f)


class PrecomputedSplitter:
    """Useful when train and test indices are pre-specified."""
    
    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(X = None, y = None):
        yield self.train_idx, self.test_idx


@logged
class Performances:
    """Calculate performances"""

    def __init__(self, is_classifier):
        self.is_classifier = is_classifier

    @staticmethod
    def truth_inv(yt):
        """Given an active (1) / inactive (0) dataset, assign 1 to inactives"""
        return np.abs(1 - yt)

    def classifier_performances(self, Yt, Yp):
        """Calculate standard prediction performance metrics.
        In addition, it calculates the corresponding weights.
        For the moment, AUROC, AUPR and BEDROC are used.
        Args:
            Yt(list of lists): Truth data (binary).
            Yp(list of lists): Prediction scores (probabilities).
        """
        self.__log.debug("Initialize variables")
        self.auroc  = []
        self.bedroc = []
        self.aupr   = []
        self.__log.debug("Calculating performances")
        for yt, yp in zip(Yt, Yp):
            shape = yp.shape
            if len(shape) == 1:
                self.__log.debug("Prediction has only one dimension, i.e. score for active")
                auroc  = metrics.roc_score(yt, yp)[0]
                aupr   = metrics.pr_score(yt, yp)[0]
                bedroc = metrics.bedroc_score(yt, yp)[0]
            else:
                assert shape[1] == 2, "Too many classes. For the moment, TargetMate only works with active/inactive (1/0)"
                assert set(yt) == set([0,1]), "Only 1/0 is accepted for the moment."
                if len(shape) == 2:
                    self.__log.debug("Prediction has two dimensions (active/inactive). Not ensemble.")
                    auroc  = np.zeros(2)
                    aupr   = np.zeros(2)
                    bedroc = np.zeros(2)
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = self.truth_inv(yt)
                    auroc[0]  = metrics.roc_score(yt_, yp[:,0])[0]
                    aupr[0]   = metrics.pr_score(yt_, yp[:,0])[0]
                    bedroc[0] = metrics.bedroc_score(yt_, yp[:,0])[0]
                    self.__log.debug("Measuring performances for actives")
                    yt_ = yt
                    auroc[1]  = metrics.roc_score(yt_, yp[:,1])[0]
                    aupr[1]   = metrics.pr_score(yt_, yp[:,1])[0]
                    bedroc[1] = metrics.bedroc_score(yt_, yp[:,1])[0]
                if len(shape) == 3:
                    self.__log.debug("Prediction has three dimensions (active/inactives and ensemble of datasets)")
                    ens_size = shape[2]
                    auroc  = np.zeros((2, ens_size))
                    aupr   = np.zeros((2, ens_size))
                    bedroc = np.zeros((2, ens_size))
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = self.truth_inv(yt)
                    for k in range(0, ens_size):
                        auroc[0, k]  = metrics.roc_score(yt_, yp[:,0,k])[0]
                        aupr[0, k]   = metrics.pr_score(yt_, yp[:,0,k])[0]
                        bedroc[0, k] = metrics.bedroc_score(yt_, yp[:,0,k])[0]
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = yt
                    for k in range(0, ens_size):
                        auroc[1, k]  = metrics.roc_score(yt_, yp[:,1,k])[0]
                        aupr[1, k]   = metrics.pr_score(yt_, yp[:,1,k])[0]
                        bedroc[1, k] = metrics.bedroc_score(yt_, yp[:,1,k])[0]
        self.auroc  += [auroc]
        self.bedroc += [bedroc]
        self.aupr   += [aupr]

    def regressor_performances(self, yt, yp):
        """TO-DO"""
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
class Gatherer:
    """Gather data"""

    def __init__(self, is_ensemble):
        self.is_ensemble  = is_ensemble
        self.train_idx    = []
        self.test_idx     = []
        self.y_true_train = []
        self.y_true_test  = []
        self.y_pred_train = []
        self.y_pred_test  = []
        if self.is_ensemble:
            self.y_pred_ens_train = []
            self.y_pred_ens_test  = []

    def gather(self, tm, data, splits):
        self.__log.info("Gathering data (splits, samples, outcomes[, datasets])")
        self.__log.debug("Iterating over splits")
        i = 0
        for tr_idx, te_idx in splits:
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=True, is_tmp=True, reset=True)
            pred_train = tm.load_predictions()
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=False, is_tmp=True, reset=True)
            pred_test  = tm.load_predictions()
            data_train = data[tr_idx]
            data_test  = data[te_idx]
            self.train_idx += [tr_idx]
            self.test_idx  += [te_idx]
            self.y_true_train += [data_train.activity]
            self.y_true_test  += [data_test.activity]
            self.y_pred_train += [pred_train.y_pred]
            self.y_pred_test  += [pred_test.y_pred]
            if self.is_ensemble:
                self.y_pred_ens_train += [pred_train.y_pred_ens]
                self.y_pred_ens_test  += [pred_test.y_pred_ens]
            i += 1


@logged
class Scorer:
    """Run performances"""

    def __init__(self, is_classifier, is_ensemble):
        self.is_classifier = is_classifier
        self.is_ensemble = is_ensemble

    def score(self, g):
        self.perfs_train = Performances(self.is_classifier).compute(g.y_true_train, g.y_pred_train)
        self.perfs_test  = Performances(self.is_classifier).compute(g.y_true_test,  g.y_pred_test )
        if self.is_ensemble:
            self.perfs_ens_train = Performances(self.is_classifier).compute(g.y_true_train, g.y_pred_ens_train)
            self.perfs_ens_test  = Performances(self.is_classifier).compute(g.y_true_test,  g.y_pred_ens_test)
        else:
            self.perfs_ens_train = None
            self.perfs_ens_test  = None


@logged
class Validation(HPCUtils):
    """Validation class."""

    def __init__(self,
                 splitter=None,
                 is_cv=True,
                 n_splits=5,
                 test_size=0.2,
                 destination_dir=None,
                 **kwargs):
        """Initialize validation class.

        Args:
            splitter(object): If none specified, the corresponding TargetMate splitter is used (default=None).
            is_cv(bool): If False, a simple train-test split is done (default=True).
            cv_folds(int): Number of CV folds (default=5).
            test_size(float): Proportion of samples in the test set (default=0.2).
            destination_dir(str): If non specified, the models path of the TargetMate instance is used (default=None).
        """
        HPCUtils.__init__(self, **kwargs)
        self.splitter = splitter
        self.is_cv = is_cv
        self.n_splits = n_splits
        self.test_size = test_size
        self.destination_dir = destination_dir
        
    def compute(self, tm, data, train_idx, test_idx):
        """Do the cross-validation"""
        # Initialize
        self.tmp_path = tm.tmp_path
        self.is_classifier = tm.is_classifier
        self.is_ensemble = tm.is_ensemble
        self.conformity = tm.conformity
        self.weights = tm.weights
        # Signaturize
        self.__log.info("Signaturizing all data")
        tm.signaturize(data.smiles, is_tmp=True)
        # Splits
        self.__log.debug("Computing validation")
        if train_idx is not None and test_idx is not None:
            kf = PrecomputedSplitter(train_idx, test_idx)
        else:
            if not self.splitter:
                if self.is_cv:
                    if tm.is_classifier:
                        kf = model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                    else:
                        self.__log.error("CV FOR REGRESSION NOT YET DONE")
                        # TO-DO
                    #kf = tm.kfolder()
                else:
                    if tm.is_classifier:
                        self.__log.info("Setting up a stratified shuffle split")
                        kf = model_selection.StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size)
                    else:
                        self.__log.error("SPLITTING FOR REGRESSION NOT YET DONE")
                        # TO-DO
            else:
                kf = self.splitter
        splits = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X=np.zeros(len(data.activity)), y=data.activity)]
        # Fit
        self.__log.info("Fitting")
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=True, reset=True)
            self.__log.info(tm.bases_tmp_path)
            jobs += tm.fit(data, idxs=train_idx, is_tmp=True, wait=False)
            i += 1
        self.waiter(jobs)
        # Predict
        self.__log.info("Predicting for train")
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=True, reset=True)
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=True, is_tmp=True, reset=True)
            self.__log.info(tm.predictions_tmp_path)
            jobs += tm.predict(data, idxs=train_idx, is_tmp=True, wait=False)
            i += 1
        self.waiter(jobs)
        self.__log.info("Predicting for test")
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=True, reset=True)
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=False, is_tmp=True, reset=True)
            self.__log.info(tm.predictions_tmp_path)
            jobs += tm.predict(data, idxs=test_idx, is_tmp=True, wait=False)
            i += 1
        self.waiter(jobs)
        self.n_splits = i
        # Gather data
        gather = Gatherer(is_ensemble = self.is_ensemble)
        gather.gather(tm, data, splits)
        self.gather_path = os.path.join(self.tmp_path, "validation-gather.pkl")
        with open(self.gather_path, "wb") as f:
            pickle.dump(gather, f)

    def score(self):
        self.__log.info("Loading gathered predictions")
        with open(self.gather_path, "rb") as f:
            gather = pickle.load(f)
        scores = Scorer(is_classifier = self.is_classifier, is_ensemble = self.is_ensemble)
        scores.score(gather)
        self.scores_path = os.path.join(self.tmp_path, "validation_scores.pkl")
        with open(self.scores_path, "wb") as f:
             pickle.dump(scores, f)
        
    def as_dict(self):
        self.__log.debug("Reading gatherer")
        with open(self.gather_path, "rb") as f:
            gather = pickle.load(f)
        self.__log.debug("Reading scores")
        with open(self.scores_path, "rb") as f:
            scores = pickle.load(f)
        self.__log.info("Converting to dictionary")
        valid = {
            "n_splits": self.n_splits,
            "dim_dict": ("splits", "molecules", "outcomes", "ensemble"),
            "is_classifier": self.is_classifier,
            "is_ensemble": self.is_ensemble,
            "conformity": self.conformity,
            "datasets": self.datasets,
            "weights": self.weights,
            "destination_dir": self.destination_dir,
            "train": {
                    "idx"   : gather.train_idx,
                    "y_true": gather.y_true_train,
                    "y_pred": gather.y_pred_train,
                    "perfs" : scores.perfs_train.as_dict()
                },
            "test": {
                    "idx"   : gather.test_idx,
                    "y_true": gather.y_true_test,
                    "y_pred": gather.y_pred_test,
                    "perfs" : scores.perfs_test.as_dict()
            }
        }
        if self.is_ensemble:
            valid["ens_train"] = {
                "y_true": gather.y_true_train,
                "y_pred": gather.y_pred_ens_train,
                "perfs" : scores.perfs_ens_train.as_dict()
            }
            valid["ens_test"]  = {
                "y_true": gather.y_true_test,
                "y_pred": gather.y_pred_ens_test,
                "perfs" : scores.perfs_ens_test.as_dict()
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

    def validate(self, tm, data, train_idx=None, test_idx=None, as_dict=True, save=True, wipe=True):
        """Validate a TargetMate model using train-test splits.

        Args:
            tm(TargetMate model): The TargetMate model to be evaluated.
            data(InputData): Data object.
            train_idx(array): Precomputed indices for the train set (default=None). 
            test_idx(array): Precomputed indices for the test set (default=None).
            as_dict(bool): Return as dictionary, for portability; if False, the validation is done inplace (default=True).
            save(bool): Save (default=True).
            wipe(bool): Clean temporary directory once done (default=True).
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
        if wipe:
            tm.wipe()
        return d

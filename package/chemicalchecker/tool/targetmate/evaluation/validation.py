"""Validate a TargetMate model (regression or classifier)"""
import os
import numpy as np
import h5py
import pickle
from chemicalchecker.util import logged
from sklearn import model_selection
from ..utils import metrics
from ..utils import HPCUtils
from ..utils import splitters

SEED = 42
MAXQUEUE = 100

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
        self.shaps_train  = []
        self.shaps_test   = []
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
            expl_train = tm.load_explanations()
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=False, is_tmp=True, reset=True)
            pred_test  = tm.load_predictions()
            expl_test  = tm.load_explanations()
            data_train = data[tr_idx]
            data_test  = data[te_idx]
            self.train_idx += [tr_idx]
            self.test_idx  += [te_idx]
            self.y_true_train += [data_train.activity]
            self.y_true_test  += [data_test.activity]
            self.y_pred_train += [pred_train.y_pred]
            self.y_pred_test  += [pred_test.y_pred]
            if expl_train:
                self.shaps_train += [expl_train.shaps]
            if expl_test:
                self.shaps_test  += [expl_test.shaps]
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
class BaseValidation(object):
    """Validation class."""

    def __init__(self,
                 splitter,
                 is_cv,
                 is_stratified,
                 n_splits,
                 test_size,
                 explain):
        """Initialize validation class.

        Args:
            splitter(object): If none specified, the corresponding TargetMate splitter is used.
            is_cv(bool): If False, a simple train-test split is done.
            is_stratified(bool): Do stratified split.
            n_splits(int): Number of splits to perform.
            test_size(float): Proportion of samples in the test set.
        """
        self.splitter       = splitter
        self.is_cv          = is_cv
        self._n_splits      = n_splits
        self.test_size      = test_size
        self.is_stratified  = is_stratified
        self.explain        = explain

        self.models_path    = []
        self.tmp_path       = []
        self.gather_path    = []
        self.scores_path    = []

        self.is_ensemble    = []
        self.is_classifier  = []
        self.conformity     = []
        self.datasets       = []
        self.weights        = []
        self.n_splits       = []

    def setup(self, tm):
        self.models_path   += [tm.models_path]
        self.tmp_path      += [tm.tmp_path]
        self.is_ensemble   += [tm.is_ensemble]
        self.is_classifier += [tm.is_classifier]
        self.conformity    += [tm.conformity]
        self.datasets      += [tm.datasets]
        self.weights       += [tm.weights]

    def get_splits(self, tm, data, train_idx, test_idx):
        self.__log.info("Splitting")
        if train_idx is not None and test_idx is not None:
            kf = PrecomputedSplitter(train_idx, test_idx)
        else:
            if not self.splitter:
                Spl = splitters.GetSplitter(is_cv=self.is_cv,
                                            is_stratified=self.is_stratified,
                                            is_classifier=tm.is_classifier,
                                            scaffold_split=tm.scaffold_split)
                kf = Spl(n_splits=self._n_splits, test_size=self.test_size, random_state=SEED)
            else:
                kf = self.splitter
        splits = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X=data.smiles, y=data.activity)]
        self.n_splits += [len(splits)]
        return splits

    def fit(self, tm, data, splits):
        self.__log.info("Fitting")
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=True, reset=True)
            self.__log.info(tm.bases_tmp_path)
            jobs += tm.fit(data, idxs=train_idx, is_tmp=True, wait=False)
            i += 1
        return jobs

    def predict(self, tm, data, splits, is_train):
        if is_train: label = "Train"
        else: label = "Test"
        self.__log.info("Predicting for %s" % label)
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            if is_train:
                idx = train_idx
            else:
                idx = test_idx
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=True, reset=True)
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=is_train, is_tmp=True, reset=True)
            self.__log.info(tm.predictions_tmp_path)
            jobs += tm.predict(data, idxs=idx, is_tmp=True, wait=False)
            if self.explain and not is_train:
                self.__log.info("Explaining (only for test)")
                jobs += tm.explain(data, idxs=idx, is_tmp=True, wait=False)
            i += 1
        return jobs
        
    def gather(self, tm, data, splits):
        # Gather data
        gather = Gatherer(is_ensemble = tm.is_ensemble)
        gather.gather(tm, data, splits)
        gather_path = os.path.join(tm.tmp_path, "validation_gather.pkl")
        with open(gather_path, "wb") as f:
            pickle.dump(gather, f)
        self.gather_path += [gather_path]

    def score(self):
        self.__log.info("Loading gathered predictions")
        for tmp_path, gather_path, is_classifier, is_ensemble in zip(self.tmp_path, self.gather_path, self.is_classifier, self.is_ensemble):
            with open(gather_path, "rb") as f:
                gather = pickle.load(f)
            scores = Scorer(is_classifier = is_classifier, is_ensemble = is_ensemble)
            scores.score(gather)
            scores_path = os.path.join(tmp_path, "validation_scores.pkl")
            with open(scores_path, "wb") as f:
                 pickle.dump(scores, f)
            self.scores_path += [scores_path]
        
    def _as_dict(self, i):
        self.__log.debug("Reading gatherer")
        with open(self.gather_path[i], "rb") as f:
            gather = pickle.load(f)
        self.__log.debug("Reading scores")
        with open(self.scores_path[i], "rb") as f:
            scores = pickle.load(f)
        self.__log.info("Converting to dictionary")
        valid = {
            "n_splits": self.n_splits[i],
            "dim_dict": ("splits", "molecules", "outcomes", "ensemble"),
            "is_classifier": self.is_classifier[i],
            "is_ensemble": self.is_ensemble[i],
            "conformity": self.conformity[i],
            "datasets": self.datasets[i],
            "weights": self.weights[i],
            "models_path": self.models_path[i],
            "train": {
                    "idx"   : gather.train_idx,
                    "y_true": gather.y_true_train,
                    "y_pred": gather.y_pred_train,
                    "shaps" : gather.shaps_train,
                    "perfs" : scores.perfs_train.as_dict()
                },
            "test": {
                    "idx"   : gather.test_idx,
                    "y_true": gather.y_true_test,
                    "y_pred": gather.y_pred_test,
                    "shaps" : gather.shaps_test,
                    "perfs" : scores.perfs_test.as_dict()
            }
        }
        if self.is_ensemble[i]:
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
        print("TRAIN AUROC", valid["train"]["perfs"]["auroc"])
        print("TEST AUROC ", valid["test"]["perfs"]["auroc"])
        return valid
    
    def as_dict(self):
        for i in range(0, len(self.models_path)):
            yield self._as_dict(i)
        
    def save(self):
        for valid in self.as_dict():
            filename = os.path.join(valid["models_path"], "validation.pkl")
            with open(filename, "wb") as f:
                pickle.dump(valid, f)


@logged
class Validation(BaseValidation, HPCUtils):

    def __init__(self,
                 splitter=None,
                 is_cv=False,
                 is_stratified=True,
                 n_splits=3,
                 test_size=0.2,
                 explain=False,
                 **kwargs):
        HPCUtils.__init__(self, **kwargs)
        BaseValidation.__init__(self, splitter, is_cv, is_stratified, n_splits, test_size, explain)

    def single_validate(self, tm, data, train_idx, test_idx, wipe, **kwargs):
        # Initialize
        self.__log.info("Setting up")
        self.setup(tm)
        # Signaturize
        self.__log.info("Signaturizing all data")
        tm.signaturize(data.smiles, is_tmp=True, wait=True)
        # Splits
        self.__log.info("Getting splits")
        splits = self.get_splits(tm, data, train_idx, test_idx)
        # Fit
        self.__log.info("Fit with train")
        jobs = self.fit(tm, data, splits)
        self.waiter(jobs)
        # Predict for train
        self.__log.info("Predict for train")
        jobs = self.predict(tm, data, splits, is_train=True)
        self.waiter(jobs)
        # Predict for test
        self.__log.info("Predict for test")
        jobs = self.predict(tm, data, splits, is_train=False)
        self.waiter(jobs)
        # Gather
        self.__log.info("Gather")
        self.gather(tm, data, splits)
        # Score
        self.__log.info("Scores")
        self.score()
        # Save
        self.__log.info("Save")
        self.save()
        # Wipe
        if wipe:
            tm.wipe()

    def multi_validate(self, tm_list, data_list, wipe, **kwargs):
        # Initialize
        self.__log.info("Setting up")
        for tm in tm_list:
            self.setup(tm)
        # Signaturize
        self.__log.info("Signaturizing all data")
        jobs = []
        for tm, data in zip(tm_list, data_list):
            jobs += tm.signaturize(data.smiles, is_tmp=True, wait=False)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Splits
        self.__log.info("Getting splits")
        splits_list = []
        for tm, data in zip(tm_list, data_list):
            splits_list += [self.get_splits(tm, data, None, None)]
        # Fit
        self.__log.info("Fit with train")
        jobs = []
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            jobs += self.fit(tm, data, splits)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Predict for train
        self.__log.info("Predict for train")
        jobs = []
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            jobs += self.predict(tm, data, splits, is_train=True)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Predict for test
        self.__log.info("Predict for test")
        jobs = []
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            jobs += self.predict(tm, data, splits, is_train=False)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Gather
        self.__log.info("Gather")
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            self.gather(tm, data, splits)
        # Score
        self.__log.info("Scores")
        self.score()
        # Save
        self.__log.info("Save")
        self.save()
        # Wipe
        if wipe:
            for tm in tm_list:
                tm.wipe()

    def validate(self, tm, data, train_idx=None, test_idx=None, wipe=True):
        """Validate a TargetMate model using train-test splits.

        Args:
            tm(TargetMate model, or list of): The TargetMate model to be evaluated. A list is accepted.
            data(InputData, or list of): Data object. A list is accepted.
            train_idx(array): Precomputed indices for the train set (default=None). 
            test_idx(array): Precomputed indices for the test set (default=None).
            wipe(bool): Clean temporary directory once done (default=True).
        """
        if type(tm) != list:
            self.single_validate(tm=tm, data=data, train_idx=train_idx, test_idx=test_idx, wipe=wipe)
        else:
            self.multi_validate(tm_list=tm, data_list=data, wipe=wipe)

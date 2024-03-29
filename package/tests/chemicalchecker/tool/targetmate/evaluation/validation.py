"""Validate a TargetMate model (regression or classifier)"""
import os
import numpy as np
import h5py
import pickle
import shutil
from chemicalchecker.util import logged
from sklearn import model_selection
from ..utils import metrics
from ..utils import HPCUtils
from ..utils import splitters
from ..io import data_from_disk, read_data
from ..ml import tm_from_disk
import joblib
from datetime import datetime

SEED = 42
MAXQUEUE = 50
MAXPARALLEL = 10


def load_validation(destination_dir):
    with open(destination_dir, "rb") as f:
        return pickle.load(f)


class PrecomputedSplitter:
    """Useful when train and test indices are pre-specified."""

    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(X=None, y=None):
        yield self.train_idx, self.test_idx


@logged
class thOpt:
    "Apply out-of-bag thresholding from ____"  # TODO: Include reference to GHOST paper

    def __init__(self, tm, is_tmp, thresholds=None):
        if thresholds is None:
            self.thresholds = np.round(np.arange(0.00, 0.550, 0.005), 3)
        else:
            self.thresholds = thresholds
        if is_tmp:
            mod_path = tm.bases_tmp_path
        else:
            mod_path = tm.bases_models_path

        mod = joblib.load(os.path.join(mod_path, "Stacked-uncalib---complete"))
        oob_probs = mod.oob_decision_function_
        self.oob_probs = [x[1] for x in oob_probs]

    def calculate(self, labels_train, ThOpt_metrics='Kappa'):
        """Optimize the decision threshold based on the prediction probabilities of the out-of-bag set of random forest.
        The threshold that maximizes the Cohen's kappa coefficient or a ROC-based criterion
        on the out-of-bag set is chosen as optimal.

        Parameters
        ----------
        oob_probs : list of floats
            Positive prediction probabilities for the out-of-bag set of a trained random forest model
        labels_train: list of int
            True labels for the training set
        thresholds: list of floats
            List of decision thresholds to screen for classification
        ThOpt_metrics: str
            Optimization metric. Choose between "Kappa" and "ROC"

        Returns
        ----------
        thresh: float
            Optimal decision threshold for classification
        """
        # Optmize the decision threshold based on the Cohen's Kappa coefficient

        if ThOpt_metrics == 'Kappa':
            tscores = []
            # evaluate the score on the oob using different thresholds
            for thresh in self.thresholds:
                scores = [1 if x >= thresh else 0 for x in self.oob_probs]
                kappa = metrics.cohen_kappa_score(labels_train, scores)
                tscores.append((np.round(kappa, 3), thresh))
            # select the threshold providing the highest kappa score as optimal
            tscores.sort(reverse=True)
            thresh = tscores[0][-1]
        # Optmize the decision threshold based on the ROC-curve
        elif ThOpt_metrics == 'ROC':
            # ROC optimization with thresholds determined by the roc_curve function of sklearn
            fpr, tpr, thresholds_roc = metrics.roc_curve(labels_train, self.oob_probs, pos_label=1)
            specificity = 1 - fpr
            roc_dist_01corner = (2 * tpr * specificity) / (tpr + specificity)
            thresh = thresholds_roc[np.argmax(roc_dist_01corner)]
        return np.float32(thresh)


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
        self.auroc = []
        self.bedroc = []
        self.aupr = []
        self.__log.debug("Calculating performances")
        for yt, yp in zip(Yt, Yp):
            shape = yp.shape
            if len(shape) == 1:
                self.__log.debug("Prediction has only one dimension, i.e. score for active")
                auroc = metrics.roc_score(yt, yp)[0]
                aupr = metrics.pr_score(yt, yp)[0]
                bedroc = metrics.bedroc_score(yt, yp)[0]
            else:
                assert shape[
                           1] == 2, "Too many classes. For the moment, TargetMate only works with active/inactive (1/0)"
                # corner case, just assign a three 1's randomly if it is all 0's.
                if set(yt) == set([0]):
                    self.__log.warn("Only zeroes in a split... randomly assigning ones")
                    idxs = np.random.choice(len(yt), 3, replace=False)
                    for i in idxs:
                        yt[i] = 1
                if len(shape) == 2:
                    self.__log.debug("Prediction has two dimensions (active/inactive). Not ensemble.")
                    auroc = np.zeros(2)
                    aupr = np.zeros(2)
                    bedroc = np.zeros(2)
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = self.truth_inv(yt)
                    auroc[0] = metrics.roc_score(yt_, yp[:, 0])[0]
                    aupr[0] = metrics.pr_score(yt_, yp[:, 0])[0]
                    bedroc[0] = metrics.bedroc_score(yt_, yp[:, 0])[0]
                    self.__log.debug("Measuring performances for actives")
                    yt_ = yt
                    auroc[1] = metrics.roc_score(yt_, yp[:, 1])[0]
                    aupr[1] = metrics.pr_score(yt_, yp[:, 1])[0]
                    bedroc[1] = metrics.bedroc_score(yt_, yp[:, 1])[0]
                if len(shape) == 3:
                    self.__log.debug("Prediction has three dimensions (active/inactives and ensemble of datasets)")
                    ens_size = shape[2]
                    auroc = np.zeros((2, ens_size))
                    aupr = np.zeros((2, ens_size))
                    bedroc = np.zeros((2, ens_size))
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = self.truth_inv(yt)
                    for k in range(0, ens_size):
                        auroc[0, k] = metrics.roc_score(yt_, yp[:, 0, k])[0]
                        aupr[0, k] = metrics.pr_score(yt_, yp[:, 0, k])[0]
                        bedroc[0, k] = metrics.bedroc_score(yt_, yp[:, 0, k])[0]
                    self.__log.debug("Measuring performance for inactives")
                    yt_ = yt
                    for k in range(0, ens_size):
                        auroc[1, k] = metrics.roc_score(yt_, yp[:, 1, k])[0]
                        aupr[1, k] = metrics.pr_score(yt_, yp[:, 1, k])[0]
                        bedroc[1, k] = metrics.bedroc_score(yt_, yp[:, 1, k])[0]
            self.auroc += [auroc]
            self.bedroc += [bedroc]
            self.aupr += [aupr]

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
                "auroc": self.auroc,
                "aupr": self.aupr,
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
        self.is_ensemble = is_ensemble
        self.train_idx = []
        self.test_idx = []
        self.putative_idx = []
        self.y_true_train = []
        self.y_true_test = []
        self.y_pred_train = []
        self.y_pred_test = []
        self.shaps_train = []
        self.shaps_test = []
        self.y_pred_train_uncalib = []
        self.y_pred_test_uncalib = []
        self.thopt_kappa = []
        self.thopt_roc = []

        if self.is_ensemble:
            self.y_pred_ens_train = []
            self.y_pred_ens_test = []

    def gather(self, tm, data, splits, is_tmp, is_tmp_bases, only_train=False, already_scored=False):
        self.__log.info("Gathering data (splits, samples, outcomes[, datasets])")
        self.__log.debug("Iterating over splits")
        self.putative_idx = tm.putative_idx

        i = 0
        for tr_idx, te_idx in splits:
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=True, is_tmp=is_tmp, reset=True, only_train = only_train)
            pred_train = tm.load_predictions()
            expl_train = tm.load_explanations()
            if not only_train:
                tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=False, is_tmp=is_tmp, reset=True)
                pred_test = tm.load_predictions()
                expl_test = tm.load_explanations()
                tm.repath_bases_by_fold(fold_number=i, is_tmp=is_tmp_bases, reset=True)
            thopt = thOpt(tm, is_tmp_bases)
            self.train_idx += [tr_idx]
            self.y_true_train += [pred_train.y_true]
            self.y_pred_train += [pred_train.y_pred.astype('float32')]
            self.y_pred_train_uncalib += [pred_train.y_pred_uncalib.astype('float32')[:,1]]
            if expl_train:
                self.shaps_train += [expl_train.shaps]
            self.thopt_kappa += [thopt.calculate(pred_train.y_true, 'Kappa')]
            self.thopt_roc += [thopt.calculate(pred_train.y_true, 'ROC')]
            if not only_train:
                self.test_idx += [te_idx]
                self.y_true_test += [pred_test.y_true]
                self.y_pred_test += [pred_test.y_pred.astype('float32')]
                self.y_pred_test_uncalib += [pred_test.y_pred_uncalib.astype('float32')[:,1]]
                if expl_test:
                    self.shaps_test += [expl_test.shaps]


            if self.is_ensemble:
                self.y_pred_ens_train += [pred_train.y_pred_ens]
                self.y_pred_ens_test += [pred_test.y_pred_ens]
            i += 1


@logged
class Scorer:
    """Run performances"""

    def __init__(self, is_classifier, is_ensemble, only_train = False):
        self.is_classifier = is_classifier
        self.is_ensemble = is_ensemble
        self.only_train = only_train

    def score(self, g):
        self.perfs_train = Performances(self.is_classifier).compute(g.y_true_train, g.y_pred_train)
        if not self.only_train:
            self.perfs_test = Performances(self.is_classifier).compute(g.y_true_test, g.y_pred_test)
        if self.is_ensemble:
            self.perfs_ens_train = Performances(self.is_classifier).compute(g.y_true_train, g.y_pred_ens_train)
            self.perfs_ens_test = Performances(self.is_classifier).compute(g.y_true_test, g.y_pred_ens_test)
        else:
            self.perfs_ens_train = None
            self.perfs_ens_test = None


@logged
class BaseValidation(object):
    """Validation class."""

    def __init__(self,
                 splitter,
                 is_cv,
                 is_stratified,
                 n_splits,
                 test_size,
                 explain,
                 model_type,
                 only_train,
                 only_validation,
                 train_size=None,
                 is_tmp=False,
                 is_tmp_bases=True,
                 is_tmp_signatures=True,
                 is_tmp_predictions=True
                 ):
        """Initialize validation class.

        Args:
            splitter(object): If none specified, the corresponding TargetMate splitter is used.
            is_cv(bool): If False, a simple train-test split is done.
            is_stratified(bool): Do stratified split.
            n_splits(int): Number of splits to perform.
            explain(bool): Calculate Shapley values for predictions.
            test_size(float): Proportion of samples in the test set.
            is_tmp(bool): Store all in temporary path
            is_tmp_bases|signatures|predictions(bool): Store bases|signatures|predictions in temporary path

        """

        self.splitter = splitter
        self.is_cv = is_cv
        self._n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.is_stratified = is_stratified
        self.explain = explain
        self.only_train = only_train
        self.only_validation = only_validation

        if is_tmp:
            self.is_tmp_bases = is_tmp
            self.is_tmp_signatures = is_tmp
            self.is_tmp_predictions = is_tmp
        else:
            self.is_tmp_bases = is_tmp_bases
            self.is_tmp_signatures = is_tmp_signatures
            self.is_tmp_predictions = is_tmp_predictions

        if (model_type == 'only_train') | (model_type == 'val_store'):
            self.is_tmp = False
            self.is_tmp_bases = False
            self.is_tmp_signatures = True
            if model_type == 'only_train':
                self.only_train = True
            elif model_type == 'val_store':
                self.only_validation = False
        elif model_type == 'val':
            pass
        elif type(model_type) == str:
            self.__log.info("Model type {:s} not available".format(model_type))
        else:
            self.__log.info("Setting model type from variables")

        self.models_path = []
        self.tmp_path = []
        self.gather_path = []
        self.scores_path = []

        self.is_ensemble = []
        self.is_classifier = []
        self.conformity = []
        self.datasets = []
        self.weights = []
        self.n_splits = []
        self.already_scored = False

    def setup(self, tm):
        self.models_path += [tm.models_path]
        self.tmp_path += [tm.tmp_path]
        self.is_ensemble += [tm.is_ensemble]
        self.is_classifier += [tm.is_classifier]
        self.conformity += [tm.conformity]
        self.datasets += [tm.datasets]
        self.weights += [tm.weights]

    def get_splits(self, tm, data, train_idx, test_idx):
        self.__log.info("Splitting")
        if train_idx is not None and test_idx is not None:
            kf = PrecomputedSplitter(train_idx, test_idx)
        else:
            if not self.splitter:
                Spl = splitters.GetSplitter(is_cv=self.is_cv,
                                            is_stratified=self.is_stratified,
                                            is_classifier=tm.is_classifier,
                                            scaffold_split=tm.scaffold_split,
                                            outofuniverse_split=tm.outofuniverse_split)
                kf = Spl(n_splits=self._n_splits,
                         test_size=self.test_size,
                         train_size=self.train_size,
                         random_state=SEED,
                         cc=tm.cc,
                         datasets=tm.outofuniverse_datasets,
                         cctype=tm.outofuniverse_cctype)
            else:
                kf = self.splitter
        if tm.outofuniverse_split:
            keys = data.inchikey
        else:
            keys = data.molecule
        splits = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X=keys, y=data.activity)]
        splits = [sp for sp in splits if sp[0] is not None and sp[1] is not None]
        self.n_splits += [len(splits)]
        return splits

    def fit(self, tm, data, splits, scramble=False):
        self.__log.info("Fitting")
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=self.is_tmp_bases, reset=True, only_train=self.only_train)
            self.__log.info(tm.bases_tmp_path)
            jobs += tm.fit(data, idxs=train_idx, is_tmp=self.is_tmp_bases, wait=False, scramble=scramble)
            i += 1
        return jobs

    def predict(self, tm, data, splits, is_train, external=False):
        if is_train:
            label = "Train"
        else:
            label = "Test"
        self.__log.info("Predicting for %s" % label)
        i = 0
        jobs = []
        for train_idx, test_idx in splits:
            if is_train:
                idx = train_idx
            else:
                idx = test_idx
            if external:
                idx = list(range(len(data.molecule)))
            self.__log.info("Fold %02d" % i)
            tm.repath_bases_by_fold(fold_number=i, is_tmp=self.is_tmp_bases, reset=True, only_train=self.only_train)
            tm.repath_predictions_by_fold_and_set(fold_number=i, is_train=is_train, is_tmp=self.is_tmp_predictions,
                                                  reset=True, only_train=self.only_train)
            self.__log.info(tm.predictions_tmp_path)
            jobs += tm.predict(data, idxs=idx, is_tmp=self.is_tmp_predictions, wait=False)
            if self.explain and not is_train:
                self.__log.info("Explaining (only for test)")
                jobs += tm.explain(data, idxs=idx, is_tmp=self.is_tmp_predictions, wait=False)
            i += 1
        return jobs

    def gather(self, tm, data, splits):
        # Gather data
        gather = Gatherer(is_ensemble=tm.is_ensemble)
        gather.gather(tm, data, splits, self.is_tmp_predictions, self.is_tmp_bases, self.only_train)
        gather_path = os.path.join(tm.tmp_path, "validation_gather.pkl")
        with open(gather_path, "wb") as f:
            pickle.dump(gather, f)
        self.gather_path += [gather_path]

    def score(self):
        self.__log.info("Loading gathered predictions")
        for tmp_path, gather_path, is_classifier, is_ensemble in zip(self.tmp_path, self.gather_path,
                                                                     self.is_classifier, self.is_ensemble):
            with open(gather_path, "rb") as f:
                gather = pickle.load(f)
            scores = Scorer(is_classifier=is_classifier, is_ensemble=is_ensemble, only_train=self.only_train)
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
            "thOpt": {"Kappa": gather.thopt_kappa,
                      "ROC": gather.thopt_roc},
            "putative_idx": gather.putative_idx,
            "train": {
                "idx": gather.train_idx,
                "y_true": gather.y_true_train,
                "y_pred": {"calibrated": gather.y_pred_train},
                "shaps": gather.shaps_train,
                "perfs": scores.perfs_train.as_dict()
            }
        }

        if not self.only_train:
            valid["test"] = {
                "idx": gather.test_idx,
                "y_true": gather.y_true_test,
                "y_pred": {"calibrated": gather.y_pred_test,
                           "uncalibrated": gather.y_pred_test_uncalib},
                "shaps": gather.shaps_test,
                "perfs": scores.perfs_test.as_dict()
            }

        if self.is_ensemble[i]:
            valid["ens_train"] = {
                "y_true": gather.y_true_train,
                "y_pred": gather.y_pred_ens_train,
                "perfs": scores.perfs_ens_train.as_dict()
            }
            valid["ens_test"] = {
                "y_true": gather.y_true_test,
                "y_pred": gather.y_pred_ens_test,
                "perfs": scores.perfs_ens_test.as_dict()
            }
        self.__log.debug("Train AUROC", valid["train"]["perfs"]["auroc"])
        if not self.only_train:
            self.__log.debug("Test  AUROC", valid["test"]["perfs"]["auroc"])
        print("Train AUROC", valid["train"]["perfs"]["auroc"])
        if not self.only_train:
            print("Test  AUROC", valid["test"]["perfs"]["auroc"])
        return valid

    def as_dict(self):
        for i in range(0, len(self.models_path)):
            yield self._as_dict(i)

    def save(self):
        for valid in self.as_dict():
            if not self.only_train:
                filename = os.path.join(valid["models_path"], "validation.pkl")
            else:
                filename = os.path.join(valid["models_path"], "only_train.pkl")
            with open(filename, "wb") as f:
                pickle.dump(valid, f)
            if not self.only_train:
                filename_txt = os.path.join(valid["models_path"], "validation.txt")
                with open(filename_txt, "w") as f:
                    f.write("Train AUROC: %s\n" % valid["train"]["perfs"]["auroc"])
                    f.write("Test  AUROC: %s\n" % valid["test"]["perfs"]["auroc"])


@logged
class Validation(BaseValidation, HPCUtils):

    def __init__(self,
                 splitter=None,
                 is_cv=False,
                 is_stratified=True,
                 n_splits=3,
                 test_size=0.2,
                 explain=False,
                 model_type=None,
                 only_train=False,
                 only_validation=True,
                 **kwargs):
        HPCUtils.__init__(self, **kwargs)
        BaseValidation.__init__(self, splitter, is_cv, is_stratified, n_splits, test_size, explain, model_type,
                 only_train,
                 only_validation)

    def single_validate(self, tm, data, train_idx, test_idx, wipe, **kwargs):
        # Initialize
        tm = tm_from_disk(tm)
        self.__log.info("Setting up")
        self.setup(tm)
        # Signaturize
        self.__log.info("Signaturizing all data")
        tm.signaturize(smiles=data.molecule, is_tmp=self.is_tmp_signatures, wait=True, moleculetype=data.moleculetype)
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

    def multi_validate(self, tm_list, data_list, wipe, scramble, **kwargs):
        # Signaturize
        self.__log.info("Signaturizing all data")
        jobs = []
        i = 0
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += tm.signaturize(smiles=data.molecule, is_tmp=self.is_tmp_signatures,
                                   wait=False,
                                   moleculetype=data.moleculetype)  # Added by Paula: Way to stack signaturizer
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
            tm.on_disk()
            i += 1
        self.waiter(jobs)
        # Splits
        self.__log.info("Getting splits")
        splits_list = []
        i = 0
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            splits_list += [self.get_splits(tm, data, None, None)]
            tm.on_disk()
            i += 1
        # Re-evaluating in light of the splits feasibility
        self.__log.info("... checking validity of splits")
        n_list = len(tm_list)
        tm_list_ = []
        data_list_ = []
        splits_list_ = []
        for i, sp in enumerate(splits_list):
            if len(sp) == 0:
                tm = tm_list[i]
                tm = tm_from_disk(tm)
                path = os.path.join(tm.models_path, "validation.txt")
                with open(path, "w") as f:
                    f.write("SPLITS NOT AVAILABLE")
            else:
                tm_list_ += [tm_list[i]]
                data_list_ += [data_list[i]]
                splits_list_ += [splits_list[i]]
        tm_list = tm_list_
        data_list = data_list_
        splits_list = splits_list_
        if len(tm_list) < n_list:
            self.__log.warn("%d of the %d desired models could be splitted" % (len(tm_list), n_list))
        # Initialize
        self.__log.info("Setting up")
        i = 0
        for tm in tm_list:
            tm = tm_from_disk(tm)
            self.setup(tm)
            tm.on_disk()
            i += 1
        # Fit
        self.__log.info("Fit with train")
        jobs = []
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += self.fit(tm, data, splits, scramble)

            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Predict for train
        self.__log.info("Predict for train")
        jobs = []
        i = 0
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += self.predict(tm, data, splits, is_train=True)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
            tm.on_disk()
            i += 0
        self.waiter(jobs)
        # Predict for test
        self.__log.info("Predict for test")
        jobs = []
        i = 0
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += self.predict(tm, data, splits, is_train=False)
            if len(jobs) > MAXQUEUE:
                print("MAXQUEUE")
                self.waiter(jobs)
                jobs = []
            tm.on_disk()
            i += 1
        self.waiter(jobs)
        # Gather
        self.__log.info("Gather")
        i = 0
        for tm, data, splits in zip(tm_list, data_list, splits_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            self.gather(tm, data, splits)
            tm.on_disk()
            i += 1
        # Score
        self.__log.info("Scores")
        self.score()
        # Save
        self.__log.info("Save")
        self.save()

        for tm in tm_list:
            tm = tm_from_disk(tm)
            tm.compress_models()

        # Wipe
        if wipe:
            self.__log.info("Wiping files")
            for tm in tm_list:
                tm = tm_from_disk(tm)
                if self.only_validation:
                    for path in os.listdir(tm.models_path):
                        if path == "validation.pkl" or path == "validation.txt" or path == "tm.pkl" or path == "trained_data.pkl":
                            continue
                        path = os.path.join(tm.models_path, path)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        if os.path.isfile(path):
                            os.remove(path)
                else:
                    for path in os.listdir(tm.models_path):
                        path = os.path.join(tm.models_path, path)
                        if os.path.isfile(path):
                            continue
                        if not os.listdir(path):
                            shutil.rmtree(path)

                self.__log.info("Wiping temporary directories")
                tm.wipe()

    def multi_validate_onlytrain(self, tm_list, data_list, wipe, scramble,
                                 **kwargs):  # Added by Paula: creates model using all data as train
        # Signaturize
        self.__log.info("Only train") # TODO: Make this more explicit
        self.__log.info("Signaturizing all data")

        jobs = []
        i = 0
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += tm.signaturize(smiles=data.molecule, is_tmp=self.is_tmp_signatures, wait=False,
                                   moleculetype=data.moleculetype)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
            tm.on_disk()
            i += 1
        self.waiter(jobs)
        self.__log.info("Collecting data - only train")

        # Initialize
        self.__log.info("Setting up")
        i = 0
        for tm in tm_list:
            tm = tm_from_disk(tm)
            self.setup(tm)
            tm.on_disk()
            self.n_splits += [None]
            i += 1

        # Fit
        self.__log.info("Fit with train")
        jobs = []
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += self.fit(tm, data, [(None, None)], scramble)  # CHANGE BACK TO TRUE ONCE CONFIRMED
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
        self.waiter(jobs)
        # Predict for train
        self.__log.info("Predict for train")
        jobs = []
        i = 0
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            jobs += self.predict(tm, data, [(None, None)], is_train=True)
            if len(jobs) > MAXQUEUE:
                self.waiter(jobs)
                jobs = []
            tm.on_disk()
            i += 0
        self.waiter(jobs)

        # Gather
        self.__log.info("Gather")
        i = 0
        for tm, data in zip(tm_list, data_list):
            tm = tm_from_disk(tm)
            data = data_from_disk(data)
            self.gather(tm, data, [(None, None)])
            tm.on_disk()
            i += 1
        # Score
        self.__log.info("Scores")
        self.score()
        # Save
        self.__log.info("Save")
        self.save()

        # Wipe
        if wipe:
            self.__log.info("Wiping files")
            for tm in tm_list:
                tm = tm_from_disk(tm)
                self.__log.info("Wiping temporary directories")
                tm.wipe()
                for path in os.listdir(tm.models_path):
                    path = os.path.join(tm.models_path, path)
                    if os.path.isfile(path):
                        continue
                    if not os.listdir(path):
                        shutil.rmtree(path)

        for tm in tm_list:
            tm = tm_from_disk(tm)
            tm.compress_models()

    def validate(self, tm, data, train_idx=None, test_idx=None, wipe=True, scramble=False,
                 set_train_idx=None, set_test_idx=None):
        """Validate a TargetMate model using train-test splits.

        Args:
            tm(TargetMate model, or list of): The TargetMate model to be evaluated. A list is accepted.
            data(InputData, or list of): Data object. A list is accepted.
            train_idx(array): Precomputed indices for the train set (default=None).
            test_idx(array): Precomputed indices for the test set (default=None).
            wipe(bool): Clean temporary directory once done (default=True).
            only_validation(bool): Only the validations files pkl and txt files are kept (default=True).
        """

        if type(tm) != list:
            self.single_validate(tm=tm, data=data, train_idx=train_idx, test_idx=test_idx, wipe=wipe)
        else:
            if self.only_train:
                self.multi_validate_onlytrain(tm_list=tm, data_list=data, wipe=wipe, scramble=scramble,
                                              set_train_idx=set_train_idx, set_test_idx=set_test_idx)
            else:
                self.multi_validate(tm_list=tm, data_list=data, wipe=wipe,
                                    scramble=scramble)

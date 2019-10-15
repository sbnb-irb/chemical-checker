"""
TargetMate base classes.
"""
import os
import shutil
import json
import h5py
import csv
import collections
import numpy as np
import pickle

import random
import math
import time
import uuid

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.stats import percentileofscore

from chemicalchecker.core import ChemicalChecker

from chemicalchecker.util import logged
from chemicalchecker.util import Config

from .utils import metrics
from .universes import Universe
from .utils import plots
from .utils import HPCUtils

@logged
class TargetMateClassifier(TargetMateSetup):
    """Set up a TargetMate classifier. It can sample negatives from a universe of molecules (e.g. ChEMBL)
    
    It does Mondrian cross-conformal prediction.
    """

    def __init__(self,
                 base_mod="naive_bayes",
                 model_config="vanilla",
                 ccp_folds=10,
                 min_class_size=10,
                 active_value=1,
                 inactive_value=None,
                 inactives_per_active=100,
                 metric="auroc",
                 universe_path=None,
                 naive_sampling=False,
                 **kwargs):
        """Set up a TargetMate classifier

        Args:
            algo(str): Base algorithm to use (see /model configuration files) (default=naive_base).
            model_config(str): Model configurations for the base classifier (default=vanilla).
            ccp_folds(int): Number of cross-conformal prediction folds. The default generator used is
                Stratified K-Folds (default=10).
            min_class_size(int): Minimum class size acceptable to train the
                classifier (default=10).
            active_value(int): When reading data, the activity value considered to be active (default=1).
            inactive_value(int): When reading data, the activity value considered to be inactive. If none specified,
                then any value different that active_value is considered to be inactive (default=None).
            inactives_per_active(int): Number of inactive to sample for each active.
                If None, only experimental actives and inactives are considered (default=100).
            metric(str): Metric to use to select the pipeline (default="auroc").
            universe_path(str): Path to the universe. If not specified, the default one is used (default=None).
            naive_sampling(bool): Sample naively (randomly), without using the OneClassSVM (default=False).
        """
        # Inherit from TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)
        # Cross-conformal folds
        self.ccp_folds = ccp_folds
        # Determine number of jobs
        if self.hpc:
            n_jobs = self.n_jobs_hpc
        else:
            n_jobs = self.n_jobs
        # Set the base classifier
        self.algo = algo
        self.model_config = model_config
        if self.model_config == "vanilla":
            from .models.vanillaconfigs import VanillaClassifierConfigs as ModConfig
            self.algo = ModConfig(self.algo, n_jobs=self.n_jobs)
        if self.model_config == "tpot":
            from .models.tpotconfigs import TPOTClassifierConfigs as ModConfig
            self.algo = ModConfig(self.algo, n_jobs=self.n_jobs)
        # Minimum size of the minority class
        self.min_class_size = min_class_size
        # Active value
        self.active_value = active_value
        # Inactive value
        self.inactive_value = inactive_value
        # Inactives per active
        self.inactives_per_active = inactives_per_active
        # Metric to use
        self.metric = metric
        # Load universe
        self.universe = Universe.load_universe(universe_path)
        # naive_sampling
        self.naive_sampling = naive_sampling
        # Others
        self.pipe = None

    def _reassemble_activity_sets(self, act, inact, putinact):
        self.__log.info("Reassembling activities. Convention: 1 = Active, -1 = Inactive, 0 = Sampled")
        data = []
        for x in list(act):
            data += [(x[1],  1, x[0], x[-1])]
        for x in list(inact):
            data += [(x[1], -1, x[0], x[-1])]
        n = np.max([x[0] for x in data]) + 1
        for i, x in enumerate(list(putinact)):
            data += [(i + n, 0, x[0], x[-1])]
        return InputData(data)

    def prepare_data(self, data):
        # Read data
        data = self.read_data(data)
        # Save training data
        self.save_data(data)
        # Sample inactives, if necessary
        actives   = set()
        inactives = set()
        for d in data:
            if d.activity == self.active_value:
                actives.update([(d.smiles, d.idx, d.inchikey)])
            else:
                if not self.inactive_value:
                    inactives.update([(d.smiles, d.idx, d.inchikey)])
                else:
                    if d.activity == self.inactive_value:
                        inactives.update([(d.smiles, d.idx, d.inchikey)])
        act, inact, putinact = self.universe.predict(actives, inactives,
                                                     inactives_per_active=self.inactives_per_active,
                                                     min_actives=self.min_class_size,
                                                     naive=self.naive_sampling)
        self.__log.info("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        self.__log.debug("Assembling and shuffling")
        data = self._reassemble_activity_sets(act, inact, putinact)
        data.shuffle()
        return data

    def prepare_for_ml(self, data):
        """Prepare data for ML, i.e. convert to 1/0 and check that there are enough samples for training"""
        self.__log.debug("Prepare for machine learning (converting to 1/0")
        # Consider putative inactives as inactives (e.g. set -1 to 0)
        self.__log.debug("Considering putative inactives as inactives for training")
        data.activity[data.activity <= 0] = 0   
        # Check that there are enough molecules for training.
        self.ny = np.sum(data.activity)
        if self.ny < self.min_class_size or (len(data.activity) - self.ny) < self.min_class_size:
            self.__log.warning(
                "Not enough valid molecules in the minority class..." +
                "Just keeping training data")
            self._is_fitted = True
            self.save()
            return
        self.__log.info("Actives %d / Merged inactives %d" % (self.ny, len(data.activity) - self.ny))
        return data

    def find_base_mod(self, X, y, destination_dir):
        """Select a pipeline, for example, using TPOT."""
        self.__log.info("Setting the base model")
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)        
        base_mod = self.algo.as_pipeline(X[shuff], y[shuff])
        if destination_dir:
            with open(destination_dir, "wb") as f:
                pickle.dump(base_mod, f)
        self.base_mod = base_mod
        self.base_mod_dir = destination_dir

    def fit(self, X, y, destination_dir):
        """Fit a model, using a specified pipeline.
        
        Args:
            X(array): Signatures matrix.
            y(array): Labels vector.
            destination_dir(str): File where to store the model.
                If not specified, the model is returned in memory (default=None).
            n_jobs(int): If jobs are specified, the number of CPUs per model are overwritten.
                This is relevant when sending jobs to the cluster (default=None).
        """
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        # Cross-conformal prediction
        self.ccp = conformal.cross_conformal_prediction(self.base_mod)
        # Fit the model
        self.ccp.fit(X[shuff], y[shuff])
        # Save the destination directory of the model
        self.ccp_dir = destination_dir
        if destination_dir:
            with open(destination_dir, "wb") as f:
                pickle.dump(self.ccp, f)

    def predict(self, X):
        """Make cross-conformal predictions.
        
        Args:
            X(array): Signatures.
        """
        idxs = np.argsort(ccp.classes)
        pred = self.ccp.predict(X)
        return pred[:,idxs]

    def kfolder(self):
        """Cross-validation splits strategy"""
        return StratifiedKFold(n_splits=int(np.min([self.cv, self.ny])),
                               shuffle=True, random_state=42)


@logged
class TargetMateRegressor(TargetMateSetup):
    """Set up a TargetMate classifier"""

    pass


@logged
class TargetMateEnsemble(TargetMateClassifier, TargetMateRegressor, Signaturizer):
    """An ensemble of models
    
    Ensemblize models in order to have a group of predictions.

    An initial step is done to select a pipeline for each dataset.
    """

    def __init__(self, is_classifier, **kwargs):
        """TargetMate ensemble classifier
        
        Args:
            is_classifier(bool): Determine if the ensemble class will be of classifier or regressor type.
        """
        if is_classifier:
            TargetMateClassifier.__init__(self, **kwargs)
        else:
            TargetMateRegressor.__init__(self, **kwargs)
        Signaturizer.__init__(self, **kwargs)
        self.pipes    = []
        self.ensemble = []

    def select_pipelines(self, data):
        """Choose a pipelines to work with, including determination of hyperparameters
        
        Args:
            data()
        """
        y = data.activity
        self.__log.info("Selecting pipelines for every CC dataset...")
        jobs = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets)):
            dest = os.path.join(self.bases_tmp_path, self.datasets[i]+".pipe")
            pipelines += [(self.datasets[i], dest)]
            if self.hpc:
                jobs += [self.func_hpc("select_pipeline", X, y, dest, self.n_jobs_hpc, cpu=self.n_jobs_hpc)]
            else:
                self.select_pipeline(X, y, destination_dir=dest)
        self.waiter(jobs)
        for pipe in pipelines:
            self.pipes += [(pipe[0], pickle.load(open(pipe[1], "rb")))]

    def fit_ensemble(self, data):
        y = data.activity
        self.__log.info("Fitting individual models (full data)")
        jobs  = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets)):
            dest = os.path.join(self.bases_models_path, self.datasets[i])
            self.ensemble += [(self.datasets[i], dest)]
            if self.hpc:
                jobs  += [self.func_hpc("fitter", X, y, dest, False, None, self.n_jobs_hpc, cpu=self.n_jobs_hpc)]
            else:
                self.fitter(X, y, destination_dir=dest)
        self.waiter(jobs)

    def ensemble_iter(self):
        for ds, dest in self.ensemble:
            yield self.load_base_model(dest)

    def metapredict(self, yp_dict, perfs, datasets=None):
        """Do meta-prediction based on dataset-specific predictions.
        Weights are given according to the performance of the individual
        predictors.
        Standard deviation across predictions is kept to estimate
        applicability domain.
        """
        if datasets is None:
            datasets = set(self.datasets)
        else:
            datasets = set(self.datasets).intersection(datasets)
        M = []
        w = []
        for dataset in datasets:
            w += [perfs[dataset]["perf_test"]
                 [self.metric][1]]  # Get the weight
            M += [yp_dict[dataset]]
        M = np.array(M)
        w = np.array(w)
        prds = []
        stds = []
        for j in range(0, M.shape[1]):
            avg, std = self.avg_and_std(M[:, j], w)
            prds += [avg]
            stds += [std]
        return np.clip(prds, 0.001, 0.999), np.clip(stds, 0.001, None)

    def individual_performances(self, cv_results):
        yts_train = cv_results["yts_train"]
        yps_train = cv_results["yps_train"]
        yts_test  = cv_results["yts_test" ]
        yps_test  = cv_results["yps_test" ]
        # Evaluate individual performances
        self.__log.info(
            "Evaluating dataset-specific performances based on the CV and" +
            "getting weights correspondingly")
        perfs = {}
        for dataset in self.datasets:
            ptrain = self.performances(yts_train, yps_train[dataset])
            ptest  = self.performances(yts_test, yps_test[dataset])
            perfs[dataset] = {"perf_train": ptrain, "perf_test": ptest}
        return perfs

    def all_performances(self, cv_results):
        """Get all performances, and do a meta-prediction.

        Returns:
            A results dictionary 
        """
        perfs = self.individual_performances(cv_results)
        # Meta-predictor on train and test data
        self.__log.info("Meta-predictions on train and test data")
        self.__log.debug("Assembling for train set")
        mps_train, std_train = self.metapredict(cv_results["yps_train"], perfs)
        self.__log.debug("Assembling for test set")
        mps_test, std_test = self.metapredict(cv_results["yps_test"], perfs)
        # Assess meta-predictor performance
        self.__log.debug("Assessing meta-predictor performance")
        ptrain = self.performances(cv_results["yts_train"], mps_train)
        ptest  = self.performances(cv_results["yts_test"], mps_test)
        perfs["MetaPred"] = {"perf_train": ptrain, "perf_test": ptest}
        results = {
            "perfs": perfs,
            "mps_train": mps_train,
            "std_train": std_train,
            "mps_test": mps_test,
            "std_test": std_test
        }
        self.save_performances(perfs)
        return results                

    def fit(self, data):
        # Use models folder
        self.is_tmp = False
        # Read data
        data = self.prepare_data(data)
        # Prepare data for machine learning
        data = self.prepare_for_ml(data)
        if not data: return
        # Signaturize
        self.signaturize(data.smiles)
        # Select pipelines
        
        

        # Fit the global classifier
        clf_ensemble = self.fit_ensemble(data)
        # Cross-validation
        cv_results = self.cross_validation(data)
        # Get performances
        ap_results = self.all_performances(cv_results)
        # 
        return
        # Finish
        self._is_fitted = True
        self._is_trained = True
        # Save the class
        self.__log.debug("Saving TargetMate instance")
        self.save()

    def predict(self, data):
        # Use temporary folder
        self.is_tmp = True


class Foo:

    def cross_conformal_prediction(self, data):
        y = data.activity
        self.__log.debug("Initializing cross-conformal scheme")
        # Initialize cross-conformal generator
        kf = self.kfolder()
        # Do the individual predictors
        self.__log.info("Training individual predictors with cross-conformal callibration")
        fold = 0
        jobs = []
        iter_info = []    
        for train_idx, calib_idx in kf.split(data.smiles, y):
            self.__log.debug("CCP fold %02d" % (fold+1))
            iter_info_ = []
            for i, dataset in enumerate(self.datasets):
                dest = os.path.join(self.bases_tmp_path, "%s-%d" % (self.datasets[i], fold))
                X_train = self.read_signature(dataset, train_idx)
                y_train = y[train_idx]
                if self.hpc:
                    jobs += [self.func_hpc("fitter", X_train, y_train, dest, True, self.pipes[i], cpu=self.n_jobs_hpc)]
                else:
                    self.fitter(X_train, y_train, dest, is_ccp=True, pipe=self.pipes[i], n_jobs=None)
                iter_info_ += [(dataset, dest)]
            iter_info += [{"fold"      : fold,
                           "train_idx" : train_idx,
                           "calib_idx" : calib_idx,
                           "iter_info_": iter_info_}]
            fold += 1
        self.waiter(jobs)
        self.__log.debug("Making predictions over the cross-conformal models")
        # Making predictions over the calibration sets
        yps_calib = collections.defaultdict(list)
        yts_calib = []
        smi_calib = []
        for info in iter_info:
            calib_idx  = info["calib_idx"]
            for dataset, dest in info["iter_info_"]:
                mod = self.load_base_model(dest)
                X_calib  = self.read_signature(dataset, calib_idx)
                yps_calib[dataset]  += self.predictor(mod, X_calib)
            yts_calib += list(y[calib_idx])
            smi_calib += list(data.smiles[calib_idx])
        results = {
            "yps_calib": yps_calib,
            "yts_calib": yts_calib,
            "smi_calib": smi_calib
        }
        return results


@logged
class TargetMateStackedClassifier(TargetMateClassifier, Signaturizer):
    pass


@logged
class TargetMateStackedRegressor(TargetMateRegressor, Signaturizer):
    pass

@logged
class ApplicabilityDomain:
    pass


@logged
class TargetMateEnsembleClassifier(TargetMateEnsemble, ApplicabilityDomain):
    """TargetMate ensemble classifier"""

    def __init__(self, **kwargs):
        """TargetMate ensemble classifier"""
        TargetMateEnsemble.__init__(self, is_classifier=True, **kwargs)

    def plot(self):
        """Plot model statistics"""
        perfs   = self.load_performances()
        ad_data = self.load_ad_data()
        plots.ensemble_classifier_plots(perfs, ad_data)


@logged
class TargetMateEnsembleRegressor(TargetMateEnsemble):
    """TargetMate ensemble regressor"""
    
    def __init__(self, **kwargs):
        TargetMateEnsemble.__init__(self, is_classifier=False, **kwargs)

    def plot(self):
        """Plot model statistics"""
        perfs   = self.load_performances()
        ad_data = self.load_ad_data()
        plots.ensemble_regressor_grid(perfs, ad_data)


"""
TargetMate base classes.
"""
import os
import collections
import numpy as np
import pickle
import random

from sklearn.base import clone
from sklearn.decomposition import PCA

from chemicalchecker.util import logged

from .universes import Universe
from .utils import metrics
from .utils import plots
from .signaturizer import Signaturizer, Fingerprinter
from .tmsetup import TargetMateRegressorSetup, TargetMateClassifierSetup
from .io import InputData, Prediction


@logged
class Model(TargetMateClassifierSetup, TargetMateRegressorSetup):
    """Generic model class"""

    def __init__(self, is_classifier, **kwargs):
        """Initialize TargetMate model.
        
        Args:
            is_classifier(bool): Is the model a classifier or a regressor?
        """
        if is_classifier:
            TargetMateClassifierSetup.__init__(self, **kwargs)
        else:
            TargetMateRegressorSetup.__init__(self, **kwargs)
        self.is_classifier = is_classifier
        self.weights = None

    def find_base_mod(self, X, y, destination_dir=None):
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

    def metric_calc(self, y_true, y_pred, metric=None):
        """Calculate metric. Returns (value, weight) tuple."""
        if not metric:
            metric = self.metric
        return metrics.Metric(metric)(y_true, y_pred)

    def _weight(self, X, y):
        """Weight the association between a certain data type and an y variable using a cross-validation scheme, and a relatively simple model."""
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        mod = self.weight_algo.as_pipeline(X[shuff], y[shuff])
        kf = self.kfolder()
        y_pred = []
        y_true = []
        for train_idx, test_idx in kf.split(X, y):
            mod.fit(X[train_idx], y[train_idx])
            if self.is_classifier:
                y_pred += list(mod.predict_proba(X[test_idx])[:,1])
            else:
                y_pred += list(mod.predict(X[test_idx]))
            y_true += list(y[test_idx])
        return self.metric_calc(np.array(y_true), np.array(y_pred))[1]

    def _fit(self, X, y, destination_dir=None):
        """Fit a model, using a specified pipeline.
        
        Args:
            X(array): Signatures matrix.
            y(array): Labels vector.
            destination_dir(str): File where to store the model.
                If not specified, the model is returned in memory (default=None).
            n_jobs(int): If jobs are specified, the number of CPUs per model are overwritten.
                This is relevant when sending jobs to the cluster (default=None).
        """
        self.find_base_mod(X, y)
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        if self.conformity:
            self.__log.info("Preparing cross-conformity")
            # Cross-conformal prediction
            self.mod = self.cross_conformal_func(self.base_mod)
        else:
            self.__log.info("Using the selected pipeline")
            self.mod = clone(self.base_mod)
        # Fit the model
        self.__log.info("Fitting")
        self.mod.fit(X[shuff], y[shuff])
        # Save the destination directory of the model
        self.mod_dir = destination_dir
        if destination_dir:
            self.__log.debug("Saving fitted model in %s" % self.mod_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(self.mod, f)

    def _predict(self, X, destination_dir=None):
        """Make predictions
    
        Returns:
            A (n_samples, n_classes) array. For now, n_classes = 2.
        """
        self.__log.info("Predicting")
        if self.conformity:
            preds = self.mod.predict(X)
        else:
            preds = self.mod.predict_proba(X)
        if destination_dir:
            self.__log.debug("Saving prediction in %s" % destination_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(preds, f)
        else:
            return preds

    def fit(self):
        return self._fit

    def predict(self):
        return self._predict


@logged
class FingerprintedModel(Model, Fingerprinter):
    """ """

    def __init__(self, **kwargs):
        """ """
        Model.__init__(self, **kwargs)
        Fingerprinter.__init__(self, do_init=False, **kwargs)
        self.is_ensemble = False


@logged
class SignaturedModel(Model, Signaturizer):

    def __init__(self, **kwargs):
        """ """
        Model.__init__(self, **kwargs)
        Signaturizer.__init__(self, do_init=False, **kwargs)

    def get_data_fit(self, data):
        data = self.prepare_data(data)
        data = self.prepare_for_ml(data)
        return data

    def get_data_predict(self, data):
        data = self.prepare_data(data)
        data = self.prepare_for_ml(data)
        return data


@logged
class StackedModel(SignaturedModel):
    """Stacked TargetMate model."""

    def __init__(self, n_components=None, **kwargs):
        """Initialize the stacked model."""
        SignaturedModel.__init__(self, **kwargs)
        self.is_ensemble = False
        self.n_components = n_components
        self.pca = None

    def pca_fit(self, X):
        if not self.n_components: return X
        if self.n_components > X.shape[1]: return X
        self.__log.debug("Doing PCA")
        self.pca = PCA(n_components = self.n_components)
        self.pca.fit(X)
        return self.pca.transform(X)

    def pca_transform(self, X):
        self.__log.debug("")
        if not self.pca: return X
        return self.pca.transform(X)

    def fit_stack(self, data, idxs):
        if idxs is None:
            y = data.activity
        else:
            y = data.activity[idxs]
        X = self.read_signatures_stacked(datasets=self.datasets, idxs=idxs)
        X = self.pca_fit(X)
        self._fit(X, y, destination_dir=None)

    def fit(self, data, idxs=None, is_tmp=False):
        """
        Fit the stacked model.

        Args:
            data(InputData): Input data.
            idxs(array): Indices to use for the signatures. If none specified, all are used (default=None).
            is_tmp(bool): Save in the temporary directory or in the models directory.
        """
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        self.fit_stack(data, idxs)

    def predict_stack(self, idxs):
        X = self.read_signatures_stacked(datasets=self.datasets, idxs=idxs)
        X = self.pca_transform(X)
        return self._predict(X)

    def predict(self, data, idxs=None, is_tmp=True):
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        return Prediction(datasets    = self.datasets,
                          y_pred      = self.predict_stack(idxs),
                          is_ensemble = self.is_ensemble,
                          weights     = None)


@logged
class EnsembleModel(SignaturedModel):
    """ """

    def __init__(self, **kwargs):
        """ """
        SignaturedModel.__init__(self, **kwargs)
        self.is_ensemble = True
        self.ensemble_dir = {}
        self.weights = {}

    def fit_ensemble(self, data, idxs, wait):
        if idxs is None:
            y = data.activity
        else:
            y = data.activity[idxs]
        jobs = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets, idxs=idxs)):
            self.__log.info("Fitting on %s" % self.datasets[i])
            if self.is_tmp:
                dest = os.path.join(self.bases_tmp_path, self.datasets[i])
            else:
                dest = os.path.join(self.bases_models_path, self.datasets[i])
            self.ensemble_dir[self.datasets[i]] = self.is_tmp
            self.weights[self.datasets[i]] = self._weight(X, y)
            if self.hpc:
                jobs += [self.func_hpc("_fit", X, y, dest, cpu=self.n_jobs_hpc)]
            else:
                self._fit(X, y, destination_dir=dest)
        if wait:
            self.waiter(jobs)
        return jobs

    def fit(self, data, idxs=None, is_tmp=False, wait=True):
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        jobs = self.fit_ensemble(data, idxs=idxs, wait=wait)
        if not wait:
            return jobs
        
    def _single_predict(self, X, dataset, dest):
        if self.ensemble_dir[dataset]:
            indiv_dest = os.path.join(self.bases_tmp_path, dataset)
        else:
            indiv_dest = os.path.join(self.bases_models_path, dataset)
        with open(indiv_dest, "rb") as f:
            self.__log.info("Reading model from %s" % indiv_dest)
            self.mod = pickle.load(f)
        self._predict(X, destination_dir = dest)

    def predict_ensemble(self, data, idxs, datasets, wait):
        datasets = self.get_datasets(datasets)
        jobs = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=datasets, idxs=idxs)):
            self.__log.info("Predicting on %s (n = %d)" % (self.datasets[i], X.shape[0]))
            if self.is_tmp:
                dest = os.path.join(self.predictions_tmp_path, self.datasets[i])
            else:
                dest = os.path.join(self.predictions_models_path, self.datasets[i])
            if self.hpc:
                jobs += [self.func_hpc("_single_predict", X, datasets[i], dest, cpu=self.n_jobs_hpc)]
            else:
                self._single_predict(X, datasets[i], dest)
        if wait:
            self.waiter(jobs)
            self.__log.info("Predictions done")
        return jobs

    def load_predictions(self, datasets=None):
        datasets = self.get_datasets(datasets)
        y_pred = []
        for i, dataset in enumerate(datasets):
            if self.is_tmp:
                dest = os.path.join(self.predictions_tmp_path, dataset)
            else:
                dest = os.path.join(self.predictions_models_path, dataset)
            with open(dest, "rb") as f:
                y_pred += [pickle.load(f)]
        y_pred = np.stack(y_pred, axis=2)
        return Prediction(datasets = datasets,
                          y_pred   = y_pred,
                          is_ensemble = self.is_ensemble,
                          weights = self.weights)

    def predict(self, data, idxs=None, datasets=None, is_tmp=True, wait=True):
        self.is_tmp = is_tmp
        datasets = self.get_datasets(datasets)
        self.signaturize(data.smiles, datasets=datasets)
        jobs = self.predict_ensemble(data, idxs=idxs, datasets=datasets, wait=wait)
        if wait:
            return self.load_predictions(datasets)
        else:
            return jobs

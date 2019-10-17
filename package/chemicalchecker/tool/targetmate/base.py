"""
TargetMate base classes.
"""
import os
import collections
import numpy as np
import pickle
import random

from sklearn.base import clone

from chemicalchecker.util import logged

from .universes import Universe
from .utils import plots
from .signaturizer import Signaturizer, Fingerprinter
from .tmsetup import TargetMateRegressorSetup, TargetMateClassifierSetup
from .io import InputData


@logged
class Model(TargetMateClassifierSetup, TargetMateRegressorSetup):
    """Generic model class"""

    def __init__(self, is_classifier, **kwargs):
        """Initialize"""
        if is_classifier:
            TargetMateClassifierSetup.__init__(self, **kwargs)
        else:
            TargetMateRegressorSetup.__init__(self, **kwargs)

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
            self.__log.debug("Saving predictions in %s" % destination_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(destination_dir)
        return preds

    def fit(self):
        return self._fit

    def predict(self):
        return self._predict


@logged
class FingerprintModel(Model, Fingerprinter):
    """ """

    def __init__(self, **kwargs):
        """ """
        Model.__init__(self, **kwargs)
        Fingerprinter.__init__(self, **kwargs)


@logged
class SignaturedModel(Model, Signaturizer):

    def __init__(self, **kwargs):
        """ """
        Model.__init__(self, **kwargs)
        Signaturizer.__init__(self, **kwargs)

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
    """ """

    def __init__(self, **kwargs):
        """ """
        SignaturedModel.__init__(self, **kwargs)

    def fit_stack(self, data):
        y = data.activity
        X = self.read_signatures_stacked(datasets=self.datasets)
        self._fit(X, y, destination_dir=None)

    def fit(self, data, is_tmp=False):
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        self.fit_stack(data)

    def predict_stack(self, data):
        X = self.read_signatures_stacked(datasets=self.datasets)
        return self._predict(X, destination_dir=None)

    def predict(self, data, is_tmp=True):
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        return self.predict_stack(data)


@logged
class EnsembleModel(SignaturedModel):
    """ """

    def __init__(self, **kwargs):
        """ """
        SignaturedModel.__init__(self, **kwargs)
        self.ensemble_dir = []

    def fit_ensemble(self, data):
        y = data.activity
        jobs = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets)):
            dest = os.path.join(self.bases_models_path, self.datasets[i])
            self.ensemble_dir += [(self.datasets[i], dest)]
            if self.hpc:
                jobs += [self.func_hpc("_fit", X, y, dest, cpu=self.n_jobs_hpc)]
            else:
                self._fit(X, y, destination_dir=dest)
        self.waiter(jobs)

    def fit(self, data):
        self.signaturize(data.smiles)
        self.fit_ensemble(data)
        
    def predict_ensemble(self, data, datasets=None):
        if not datasets:
            datasets = self.datasets
        else:
            datasets = sorted(set(datasets).intersection(self.datasets))

        jobs = []
        for i, X in enumerate(self.read_signatures_ensemble(datasets=datasets)):
            dest = os.path.join(self.bases_tmp_path, datasets[i])
            if self.hpc:
                jobs += [self.func_hpc("_predict", X, dest, cpu=self.n_jobs_hpc)]
            else:
                self._predict(X)

    def predict(self, data, datasets=None, is_tmp=True):
        self.is_tmp = is_tmp
        self.signaturize(data.smiles)
        for i, X in enumerate(self.read_signatures_ensemble(datasets=datasets)):
            pass



# @logged
# class TargetMateEnsemble(TargetMateClassifier, TargetMateRegressor, Signaturizer):
#     """An ensemble of models
    
#     Ensemblize models in order to have a group of predictions.

#     An initial step is done to select a pipeline for each dataset.
#     """

#     def __init__(self, is_classifier, **kwargs):
#         """TargetMate ensemble classifier
        
#         Args:
#             is_classifier(bool): Determine if the ensemble class will be of classifier or regressor type.
#         """
#         if is_classifier:
#             TargetMateClassifier.__init__(self, **kwargs)
#         else:
#             TargetMateRegressor.__init__(self, **kwargs)
#         Signaturizer.__init__(self, **kwargs)
#         self.pipes    = []
#         self.ensemble = []

#     def select_pipelines(self, data):
#         """Choose a pipelines to work with, including determination of hyperparameters
        
#         Args:
#             data()
#         """
#         y = data.activity
#         self.__log.info("Selecting pipelines for every CC dataset...")
#         jobs = []
#         for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets)):
#             dest = os.path.join(self.bases_tmp_path, self.datasets[i]+".pipe")
#             pipelines += [(self.datasets[i], dest)]
#             if self.hpc:
#                 jobs += [self.func_hpc("select_pipeline", X, y, dest, self.n_jobs_hpc, cpu=self.n_jobs_hpc)]
#             else:
#                 self.select_pipeline(X, y, destination_dir=dest)
#         self.waiter(jobs)
#         for pipe in pipelines:
#             self.pipes += [(pipe[0], pickle.load(open(pipe[1], "rb")))]

#     def fit_ensemble(self, data):
#         y = data.activity
#         self.__log.info("Fitting individual models (full data)")
#         jobs  = []
#         for i, X in enumerate(self.read_signatures_ensemble(datasets=self.datasets)):
#             dest = os.path.join(self.bases_models_path, self.datasets[i])
#             self.ensemble += [(self.datasets[i], dest)]
#             if self.hpc:
#                 jobs  += [self.func_hpc("fit_xy", X, y, dest, False, None, self.n_jobs_hpc, cpu=self.n_jobs_hpc)]
#             else:
#                 self.fit_xy(X, y, destination_dir=dest)
#         self.waiter(jobs)

#     def ensemble_iter(self):
#         for ds, dest in self.ensemble:
#             yield self.load_base_model(dest)

#     def metapredict(self, yp_dict, perfs, datasets=None):
#         """Do meta-prediction based on dataset-specific predictions.
#         Weights are given according to the performance of the individual
#         predictors.
#         Standard deviation across predictions is kept to estimate
#         applicability domain.
#         """
#         if datasets is None:
#             datasets = set(self.datasets)
#         else:
#             datasets = set(self.datasets).intersection(datasets)
#         M = []
#         w = []
#         for dataset in datasets:
#             w += [perfs[dataset]["perf_test"]
#                  [self.metric][1]]  # Get the weight
#             M += [yp_dict[dataset]]
#         M = np.array(M)
#         w = np.array(w)
#         prds = []
#         stds = []
#         for j in range(0, M.shape[1]):
#             avg, std = self.avg_and_std(M[:, j], w)
#             prds += [avg]
#             stds += [std]
#         return np.clip(prds, 0.001, 0.999), np.clip(stds, 0.001, None)

#     def individual_performances(self, cv_results):
#         yts_train = cv_results["yts_train"]
#         yps_train = cv_results["yps_train"]
#         yts_test  = cv_results["yts_test" ]
#         yps_test  = cv_results["yps_test" ]
#         # Evaluate individual performances
#         self.__log.info(
#             "Evaluating dataset-specific performances based on the CV and" +
#             "getting weights correspondingly")
#         perfs = {}
#         for dataset in self.datasets:
#             ptrain = self.performances(yts_train, yps_train[dataset])
#             ptest  = self.performances(yts_test, yps_test[dataset])
#             perfs[dataset] = {"perf_train": ptrain, "perf_test": ptest}
#         return perfs

#     def all_performances(self, cv_results):
#         """Get all performances, and do a meta-prediction.

#         Returns:
#             A results dictionary 
#         """
#         perfs = self.individual_performances(cv_results)
#         # Meta-predictor on train and test data
#         self.__log.info("Meta-predictions on train and test data")
#         self.__log.debug("Assembling for train set")
#         mps_train, std_train = self.metapredict(cv_results["yps_train"], perfs)
#         self.__log.debug("Assembling for test set")
#         mps_test, std_test = self.metapredict(cv_results["yps_test"], perfs)
#         # Assess meta-predictor performance
#         self.__log.debug("Assessing meta-predictor performance")
#         ptrain = self.performances(cv_results["yts_train"], mps_train)
#         ptest  = self.performances(cv_results["yts_test"], mps_test)
#         perfs["MetaPred"] = {"perf_train": ptrain, "perf_test": ptest}
#         results = {
#             "perfs": perfs,
#             "mps_train": mps_train,
#             "std_train": std_train,
#             "mps_test": mps_test,
#             "std_test": std_test
#         }
#         self.save_performances(perfs)
#         return results                

#     def fit(self, data):
#         # Use models folder
#         self.is_tmp = False
#         # Read data
#         data = self.prepare_data(data)
#         # Prepare data for machine learning
#         data = self.prepare_for_ml(data)
#         if not data: return
#         # Signaturize
#         self.signaturize(data.smiles)
#         # Select pipelines
        
        

#         # Fit the global classifier
#         clf_ensemble = self.fit_ensemble(data)
#         # Cross-validation
#         cv_results = self.cross_validation(data)
#         # Get performances
#         ap_results = self.all_performances(cv_results)
#         # 
#         return
#         # Finish
#         self._is_fitted = True
#         self._is_trained = True
#         # Save the class
#         self.__log.debug("Saving TargetMate instance")
#         self.save()

#     def predict(self, data):
#         # Use temporary folder
#         self.is_tmp = True


# class Foo:

#     def cross_conformal_prediction(self, data):
#         y = data.activity
#         self.__log.debug("Initializing cross-conformal scheme")
#         # Initialize cross-conformal generator
#         kf = self.kfolder()
#         # Do the individual predictors
#         self.__log.info("Training individual predictors with cross-conformal callibration")
#         fold = 0
#         jobs = []
#         iter_info = []    
#         for train_idx, calib_idx in kf.split(data.smiles, y):
#             self.__log.debug("CCP fold %02d" % (fold+1))
#             iter_info_ = []
#             for i, dataset in enumerate(self.datasets):
#                 dest = os.path.join(self.bases_tmp_path, "%s-%d" % (self.datasets[i], fold))
#                 X_train = self.read_signature(dataset, train_idx)
#                 y_train = y[train_idx]
#                 if self.hpc:
#                     jobs += [self.func_hpc("fitter", X_train, y_train, dest, True, self.pipes[i], cpu=self.n_jobs_hpc)]
#                 else:
#                     self.fitter(X_train, y_train, dest, is_ccp=True, pipe=self.pipes[i], n_jobs=None)
#                 iter_info_ += [(dataset, dest)]
#             iter_info += [{"fold"      : fold,
#                            "train_idx" : train_idx,
#                            "calib_idx" : calib_idx,
#                            "iter_info_": iter_info_}]
#             fold += 1
#         self.waiter(jobs)
#         self.__log.debug("Making predictions over the cross-conformal models")
#         # Making predictions over the calibration sets
#         yps_calib = collections.defaultdict(list)
#         yts_calib = []
#         smi_calib = []
#         for info in iter_info:
#             calib_idx  = info["calib_idx"]
#             for dataset, dest in info["iter_info_"]:
#                 mod = self.load_base_model(dest)
#                 X_calib  = self.read_signature(dataset, calib_idx)
#                 yps_calib[dataset]  += self.predictor(mod, X_calib)
#             yts_calib += list(y[calib_idx])
#             smi_calib += list(data.smiles[calib_idx])
#         results = {
#             "yps_calib": yps_calib,
#             "yts_calib": yts_calib,
#             "smi_calib": smi_calib
#         }
#         return results


# @logged
# class TargetMateStackedClassifier(TargetMateClassifier, Signaturizer):
#     pass


# @logged
# class TargetMateStackedRegressor(TargetMateRegressor, Signaturizer):
#     pass

# @logged
# class ApplicabilityDomain:
#     pass


# @logged
# class TargetMateEnsembleClassifier(TargetMateEnsemble, ApplicabilityDomain):
#     """TargetMate ensemble classifier"""

#     def __init__(self, **kwargs):
#         """TargetMate ensemble classifier"""
#         TargetMateEnsemble.__init__(self, is_classifier=True, **kwargs)

#     def plot(self):
#         """Plot model statistics"""
#         perfs   = self.load_performances()
#         ad_data = self.load_ad_data()
#         plots.ensemble_classifier_plots(perfs, ad_data)


# @logged
# class TargetMateEnsembleRegressor(TargetMateEnsemble):
#     """TargetMate ensemble regressor"""
    
#     def __init__(self, **kwargs):
#         TargetMateEnsemble.__init__(self, is_classifier=False, **kwargs)

#     def plot(self):
#         """Plot model statistics"""
#         perfs   = self.load_performances()
#         ad_data = self.load_ad_data()
#         plots.ensemble_regressor_grid(perfs, ad_data)


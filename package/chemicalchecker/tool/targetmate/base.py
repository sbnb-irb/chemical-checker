"""
TargetMate base classes.
"""
import os
import pickle
import uuid

import joblib
import numpy as np
import shap
from chemicalchecker.util import logged
from sklearn.decomposition import PCA

from .io import InputData, Prediction, Explanation
from .signaturizer import SignaturizerSetup
from .tmsetup import ModelSetup
from .utils import metrics
from .utils import splitters


@logged
class Model(ModelSetup):
    """Generic model class"""

    def __init__(self, is_classifier, **kwargs):
        """Initialize TargetMate model.

        Args:
            is_classifier(bool): Is the model a classifier or a regressor?
        """
        ModelSetup.__init__(self, is_classifier, **kwargs)
        self.weights = None
        self.mod_dir = None
        self.mod_uncalib_dir = None

    def array_on_disk(self, ar):
        fn = os.path.join(self.arrays_tmp_path, str(uuid.uuid4()) + ".npy")
        np.save(fn, ar)
        return fn

    def check_array_from_disk(self, ar):
        if ar is None:
            return None
        if type(ar) == str:
            return np.load(ar, allow_pickle=True)
        else:
            return ar

    def sampler(self, X, y):
        self.shuffle = True
        spl = splitters.ToppedSampler(max_samples=self.max_train_samples,
                                      max_ensemble_size=self.max_train_ensemble,
                                      chance=self.train_sample_chance,
                                      try_balance=True,
                                      shuffle=self.shuffle,
                                      brute=True)
        self.__log.debug("Sampling")
        for shuff in spl.sample(X=X, y=y):
            yield shuff

    def find_base_mod(self, X, y, smiles, destination_dir):
        """Select a pipeline, for example, using HyperOpt."""
        destdir = destination_dir + "---base_model"
        X = self.check_array_from_disk(X)
        y = self.check_array_from_disk(y)
        smiles = self.check_array_from_disk(smiles)
        self.__log.info("Setting the base model")
        for shuff in self.sampler(X, y):
            break
        base_mod = self.algo.as_pipeline(X=X[shuff], y=y[shuff], smiles=smiles[shuff])
        self.__log.info("Saving base model in %s" % destdir)
        self.base_mod_dir = destdir
        with open(self.base_mod_dir, "wb") as f:
            joblib.dump(base_mod, f)

    def load_base_mod(self):
        """Load base model"""
        with open(self.base_mod_dir, "rb") as f:
            return joblib.load(f)

    def metric_calc(self, y_true, y_pred, metric=None):
        """Calculate metric. Returns (value, weight) tuple."""
        if not metric:
            metric = self.metric
        return metrics.Metric(metric)(y_true, y_pred)

    def _weight(self, X, y):
        """Weight the association between a certain data type and an y variable using a cross-validation scheme, and a relatively simple model."""
        X = self.check_array_from_disk(X)
        y = self.check_array_from_disk(y)
        for shuff in self.sampler(X, y):
            break
        X = X[shuff]
        y = y[shuff]
        mod = self.weight_algo.as_pipeline(X, y)
        Spl = splitters.GetSplitter(is_cv=False,
                                    is_classifier=self.is_classifier,
                                    is_stratified=True,
                                    scaffold_split=self.scaffold_split)
        kf = Spl(n_splits=1, test_size=0.2, random_state=42)
        y_pred = []
        y_true = []
        for train_idx, test_idx in kf.split(X, y):
            mod.fit(X[train_idx], y[train_idx])
            if self.is_classifier:
                y_pred += list(mod.predict_proba(X[test_idx])[:, 1])
            else:
                y_pred += list(mod.predict(X[test_idx]))
            y_true += list(y[test_idx])
        return self.metric_calc(np.array(y_true), np.array(y_pred))[1]

    def _check_y(self, y):
        """Randomly sample positives or negatives if not enough in the class. Should be a corner case."""
        y = np.array(y)
        act = np.sum(y)
        ina = len(y) - act
        if act >= self.min_class_size_active and ina >= self.min_class_size_inactive:  # Added by Paula: Specific to each class
            return y
        if act < self.min_class_size_active:  # Added by Paula: Specific to each class
            m = self.min_class_size_active - act
            idxs = np.argwhere(y == 0).ravel()
            np.random.shuffle(idxs)
            idxs = idxs[:m]
            y[idxs] = 1
            return y
        if ina < self.min_class_size_inactive:  # Added by Paula: Specific to each class
            m = self.min_class_size_inactive - ina
            idxs = np.argwhere(y == 1).ravel()
            np.random.shuffle(idxs)
            idxs = idxs[:m]
            y[idxs] = 0
            return y

    def _fit(self, X, y, smiles=None, destination_dir=None):
        """Fit a model, using a specified pipeline.

        Args:
            X(array): Signatures matrix.
            y(array): Labels vector.
            destination_dir(str): File where to store the model.
                If not specified, the model is returned in memory (default=None).
            n_jobs(int): If jobs are specified, the number of CPUs per model are overwritten.
                This is relevant when sending jobs to the cluster (default=None).
        """
        # Set up
        if destination_dir is None:
            raise Exception("destination_dir cannot be None")
        self.mod_dir = destination_dir
        self.mod_uncalib_dir = destination_dir + "-uncalib"
        # Check if array is a file
        X = self.check_array_from_disk(X)
        y = self.check_array_from_disk(y)
        smiles = self.check_array_from_disk(smiles)
        # Get started
        self.find_base_mod(X, self._check_y(y), smiles, destination_dir=destination_dir)
        base_mod = self.load_base_mod()
        for i, shuff in enumerate(self.sampler(X, y)):
            X_ = X[shuff]
            y_ = y[shuff]
            y_ = self._check_y(y_)
            self.__log.info("Fitting round %i" % i)
            if self.conformity:
                n_models = min(self.ccp_folds, np.sum(y_))
                self.__log.info("Preparing cross-conformity (n_models = %d)" % n_models)
                # Cross-conformal prediction
                mod = self.cross_conformal_func(base_mod, n_models=n_models)
                # Do normal as well
                mod_uncalib = base_mod
            else:
                self.__log.info("Using the selected pipeline")
                mod = base_mod
                mod_uncalib = None
            # Fit the model
            self.__log.info("Fitting (%d actives, %d inactives)" % (np.sum(y_), len(shuff) - np.sum(y_)))
            mod.fit(X_, y_)
            if mod_uncalib is not None:
                self.__log.info("Fitting model, but without calibration")
                mod_uncalib.fit(X_, y_)
            for p in range(len(mod.predictors)): # Added by Paula: way to reduce memory when storing models
                mod.predictors[p].cal_x = None
                mod.predictors[p].cal_y = None
            # Save the destination directory of the model
            destdir = self.mod_dir + "---%d" % i
            destdir_uncalib = self.mod_uncalib_dir + "---%d" % i
            self.__log.debug("Saving fitted model in %s" % destdir)

            with open(destdir, "wb") as f:
                joblib.dump(mod, f)
            if mod_uncalib is not None:
                self.__log.debug("Saving fitted uncalibrated model in %s" % destdir_uncalib)
                with open(destdir_uncalib, "wb") as f:
                    joblib.dump(mod_uncalib, f)
            else:
                self.__log.debug("Calibrated and uncalibrated models are the same %s" % destdir_uncalib)
                os.symlink(destdir, destdir_uncalib)
        self.__log.info("Fitting full model, but without calibration")

        if type(mod_uncalib).__name__ == 'RandomForestClassifier':
            mod_uncalib.oob_score = True
            mod_uncalib.fit(X, y)
            destdir_uncalib = self.mod_uncalib_dir + "---complete"
            with open(destdir_uncalib, "wb") as f:
                joblib.dump(mod_uncalib, f)

    def model_iterator(self, uncalib):
        if uncalib:
            mod_dir = self.mod_uncalib_dir
        else:
            mod_dir = self.mod_dir
        name = mod_dir.split("/")[-1]
        path = os.path.dirname(mod_dir)
        for m in os.listdir(path):
            if m.split("---")[0] != name: continue
            if m.split("---")[-1] == "base_model": continue
            fn = os.path.join(path, m)
            with open(fn, "rb") as f:
                mod = joblib.load(fn)
            yield mod

    def _predict(self, X, destination_dir=None):
        """Make predictions

        Returns:
            A (n_samples, n_classes) array. For now, n_classes = 2.
        """
        X = self.check_array_from_disk(X)
        self.__log.info("Predicting")
        preds = None
        smoothing = None
        for i, mod in enumerate(self.model_iterator(uncalib=False)):
            if smoothing is None:
                smoothing = np.random.uniform(0, 1, size=(len(X), len(mod.classes)))  # Added by Paula: Apply same smoothing to all ccp
            if self.conformity:
                p = mod.predict(X, smoothing = smoothing)
            else:
                p = mod.predict_proba(X)
            if preds is None:
                preds = p
            else:
                preds = preds + p
        preds = preds / (i + 1)
        if destination_dir:
            self.__log.debug("Saving prediction in %s" % destination_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(preds, f)
        else:
            return preds

    def _predict_uncalib(self, X, destination_dir=None):
        """Make predictions

        Returns:
            A (n_samples, n_classes) array. For now, n_classes = 2.
        """
        X = self.check_array_from_disk(X)
        self.__log.info("Predicting")
        path = self.mod_uncalib_dir + "---complete"

        with open(path, "rb") as f:
            mod = joblib.load(f)
        preds = mod.predict_proba(X)
        if destination_dir:
            self.__log.debug("Saving uncalibrated prediction in %s" % destination_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(preds, f)
        else:
            return preds

    def _explain(self, X, destination_dir=None):
        """Explain the output of a model.

        Returns:
        """
        X = self.check_array_from_disk(X)
        self.__log.info("Explaining")
        shaps = None
        for i, mod in enumerate(self.model_iterator(uncalib=True)):
            explainer = shap.TreeExplainer(
                mod)  # TO-DO: Apply kernel explainer for non-tree methods. Perhaps use LIME when computational cost is high.
            shaps = explainer.shap_values(X, check_additivity=False)
            break
        if destination_dir:
            self.__log.debug("Saving explanations in %s" % destination_dir)
            with open(destination_dir, "wb") as f:
                pickle.dump(shaps, f)
        else:
            return shaps

    def fit(self):
        return self._fit

    def predict(self):
        return self._predict

    def explain(self):
        return self._explain


@logged
class SignaturedModel(Model, SignaturizerSetup): ## Commented by Paula
# class SignaturedModel(Model, Fingerprinter, Signaturizer):

    def __init__(self, **kwargs):
        """Initialize a signatured model."""
        Model.__init__(self, **kwargs)
        SignaturizerSetup.__init__(self, do_init=False, **kwargs)

    def get_data_fit(self, data, inchikey_idx=None, activity_idx=0, srcid_idx=None, use_inchikey=False, **kwargs):

        smiles_idx = kwargs.get('smiles_idx', None)
        inchi_idx = kwargs.get('inchi_idx', None)
        data = self.prepare_data(data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey)

        data = self.prepare_for_ml(data)

        return data

    def get_data_predict(self, data, smiles_idx=None, inchi_idx=None, inchikey_idx=None, activity_idx=None,
                         srcid_idx=None, use_inchikey=False, **kwargs):
        data = self.prepare_data(data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey)
        data = self.prepare_for_ml(data, predict=True)  # TODO: Check that this is necessary...
        return data

    def get_Xy_from_data(self, data, idxs, scramble = False):
        """Given data and certain idxs, get X and y (if available)"""
        # filter data by specified indices
        res = data.as_dict(idxs)
        y = res["activity"]
        idxs = res["idx"]
        molecule = res["molecule"]
        inchikeys = res["inchikey"]
        X, idxs_ = self.read_signatures(datasets=self.datasets, idxs=idxs,
                                        smiles=molecule, inchikeys=inchikeys, is_tmp=self.is_tmp_signatures)

        if y is not None:
            y = y[idxs_]
            if scramble:
                self.__log.info("Scrambling y")
                np.random.shuffle(y)
        idxs = idxs[idxs_]
        molecule = molecule[idxs_]
        inchikeys = inchikeys[idxs_]
        self.__log.info("X shape: (%d, %d) / Y length: %d" % (X.shape[0], X.shape[1], len(y)))
        # saving arrays on disk
        X = self.array_on_disk(X)
        if y is not None:
            y = self.array_on_disk(y)
        molecule = self.array_on_disk(molecule)
        inchikeys = self.array_on_disk(inchikeys)
        idxs = self.array_on_disk(idxs)
        results = {
            "X": X,
            "y": y,
            "idxs": idxs,
            "molecule": molecule,
            "inchikeys": inchikeys
        }
        self.__log.info("Arrays saved on disk %s" % results)
        return results


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
        if not self.n_components: return
        if self.n_components > X.shape[1]: return
        self.__log.debug("Doing PCA fit")
        for shuff in self.sampler(X, y=None):
            break
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X[shuff])

    def pca_transform(self, X):
        self.__log.debug("Doing PCA transform")
        if not self.pca: return X
        return self.pca.transform(X)

    def _prepare_fit_stack(self, data, idxs, scramble):
        self.__log.info("Reading signatures from data")
        res = self.get_Xy_from_data(data, idxs, scramble)
        X = res["X"]
        y = res["y"]

        molecule = res["molecule"]
        if self.is_tmp_bases:
            dest = os.path.join(self.bases_tmp_path, "Stacked")
        else:
            dest = os.path.join(self.bases_models_path, "Stacked")
        return X, y, molecule, dest

    def _fit_stack(self, data, idxs, scramble):
        X, y, smiles, dest = self._prepare_fit_stack(data, idxs, scramble)
        return self._fit(X, y, smiles=smiles, destination_dir=dest)

    def fit_stack(self, data, idxs, wait, scramble):
        jobs = []
        if self.use_cc:
            self.__log.info("Reading and fitting altogether")
            if self.hpc:
                jobs += [self.func_hpc("_fit_stack", data, idxs, scramble, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._fit_stack(data, idxs, scramble)
        else:
            X, y, smiles, dest = self._prepare_fit_stack(data, idxs, scramble)
            if self.hpc:
                jobs += [self.func_hpc("_fit", X, y, smiles, dest, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._fit(X, y, smiles=smiles, destination_dir=dest)
        if wait:
            self.waiter(jobs)
        return jobs

    def fit(self, data, idxs=None, is_tmp=False, wait=True, scramble=False):
        """
        Fit the stacked model.

        Args:
            data(InputData): Input data.
            idxs(array): Indices to use for the signatures. If none specified, all are used (default=None).
            is_tmp(bool): Save in the temporary directory or in the models directory.
        """

        jobs = self.fit_stack(data, idxs=idxs, wait=wait, scramble = scramble)
        if not wait:
            return jobs

    def _predict_(self, X, dest, dest_uncalib):
        self.__log.info("saving paths for stacked")
        if self.is_tmp_bases:
            dest_ = os.path.join(self.bases_tmp_path, "Stacked")
            dest_uncalib_ = os.path.join(self.bases_tmp_path, "Stacked-uncalib")
        else:
            dest_ = os.path.join(self.bases_models_path, "Stacked")
            dest_uncalib_ = os.path.join(self.bases_models_path, "Stacked-uncalib")
        self.mod_dir = dest_
        self.mod_uncalib_dir = dest_uncalib_
        self._predict(X, dest)
        self._predict_uncalib(X, dest_uncalib)

    def _prepare_predict_stack(self, data, idxs):
        self.__log.info("Reading signatures from data")
        res = self.get_Xy_from_data(data, idxs)
        X = res["X"]
        y = res["y"]
        # X = self.pca_transform(X)

        if self.is_tmp_predictions:
            dest = os.path.join(self.predictions_tmp_path, "Stacked")
            dest_uncalib = os.path.join(self.predictions_tmp_path, "Stacked_uncalib")
        else:
            dest = os.path.join(self.predictions_models_path, "Stacked")
            dest_uncalib = os.path.join(self.predictions_models_path, "Stacked_uncalib")
        self.__log.info("Saving metadata in %s-meta" % dest)
        meta = {
            "y": self.check_array_from_disk(y),
            "idxs": idxs,
        }
        with open(dest + "-meta", "wb") as f:
            pickle.dump(meta, f)
        return X, dest, dest_uncalib

    def _predict_stack(self, data, idxs):
        X, dest, dest_uncalib = self._prepare_predict_stack(data, idxs)
        return self._predict_(X, dest, dest_uncalib)

    def predict_stack(self, data, idxs, wait):
        jobs = []
        if self.use_cc:
            if self.hpc:
                jobs += [self.func_hpc("_predict_stack", data, idxs, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._predict_stack(data, idxs)
        else:
            X, dest, dest_uncalib = self._prepare_predict_stack(data, idxs)
            self.__log.info("Starting predictions")
            jobs = []
            if self.hpc:
                jobs += [self.func_hpc("_predict_", X, dest, dest_uncalib, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._predict_(X, dest, dest_uncalib)
        if wait:
            self.waiter(jobs)
        return jobs

    def load_predictions(self, datasets=None):
        datasets = self.get_datasets(datasets)
        # y_pred_calibrated = []
        #
        if self.is_tmp_predictions:
            dest = os.path.join(self.predictions_tmp_path, "Stacked")
            dest_uncalib = os.path.join(self.predictions_tmp_path, "Stacked_uncalib")
        else:
            dest = os.path.join(self.predictions_models_path, "Stacked")
            dest_uncalib = os.path.join(self.predictions_models_path, "Stacked_uncalib")

        with open(dest, "rb") as f:
            y_pred_calibrated = pickle.load(f)
        with open(dest_uncalib, "rb") as f:
            y_pred_uncalibrated = pickle.load(f)
        with open(dest + "-meta", "rb") as f:
            meta = pickle.load(f)
            y_true = meta["y"]

        return Prediction(
            datasets=datasets,
            y_true=y_true,
            y_pred_calibrated=y_pred_calibrated,
            y_pred_uncalibrated=y_pred_uncalibrated,
            is_ensemble=self.is_ensemble,
            weights=None
        )

    def predict(self, data, idxs=None, datasets=None, is_tmp=True, wait=True):
        # self.is_tmp = is_tmp
        datasets = self.get_datasets(datasets)
        # self.signaturize(smiles=data.molecule, datasets=datasets, moleculetype=data.moleculetype)
        jobs = self.predict_stack(data, idxs=idxs, wait=wait)
        if wait:
            return self.load_predictions(datasets)
        else:
            return jobs

    def _explain_(self, X, dest):
        if self.is_tmp_bases:
            dest_ = os.path.join(self.bases_tmp_path, "Stacked")
        else:
            dest_ = os.path.join(self.bases_models_path, "Stacked")
        if self.conformity:
            self.mod_uncalib_dir = dest_ + "-uncalib"
        else:
            self.mod_dir = dest_
            self.mod_uncalib_dir = dest_ + "-uncalib"
        self._explain(X, dest)

    def _prepare_explain_stack(self, data, idxs):
        self.__log.info("Getting signatures from data")
        res = self.get_Xy_from_data(data, idxs)
        X = res["X"]
        # X = self.pca_transform(X)
        if self.is_tmp_predictions:
            dest = os.path.join(self.predictions_tmp_path, "Stacked-expl")
        else:
            dest = os.path.join(self.predictions_models_path, "Stacked-expl")
        return X, dest

    def _explain_stack(self, data, idxs):
        X, dest = self._prepare_explain_stack(data, idxs)
        self._explain_(X, dest)

    def explain_stack(self, data, idxs, wait):
        jobs = []
        if self.use_cc:
            if self.hpc:
                jobs += [self.func_hpc("_explain_stack", data, idxs, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._explain_stack(data, idxs)
        else:
            X, dest = self._prepare_explain_stack(data, idxs)
            if self.hpc:
                jobs += [self.func_hpc("_explain_", X, dest, cpu=self.n_jobs_hpc, job_base_path=self.tmp_path)]
            else:
                self._explain_(X, dest)
        if wait:
            self.waiter(jobs)
        return jobs

    def load_explanations(self, datasets=None):
        datasets = self.get_datasets(datasets)
        y_pred = []
        if self.is_tmp_predictions:
            dest = os.path.join(self.predictions_tmp_path, "Stacked-expl")
        else:
            dest = os.path.join(self.predictions_models_path, "Stacked-expl")
        if not os.path.exists(dest):
            self.__log.info("Explanations not found")
            return None
        with open(dest, "rb") as f:
            shaps = pickle.load(f)
        return Explanation(
            datasets=datasets,
            shaps=shaps,
            is_ensemble=self.is_ensemble
        )

    def explain(self, data, idxs=None, datasets=None, is_tmp=True, wait=True):
        # self.is_tmp = is_tmp
        datasets = self.get_datasets(datasets)
        self.signaturize(smiles=data.molecule, datasets=datasets, moleculetype=data.moleculetype)
        jobs = self.explain_stack(data, idxs=idxs, wait=wait)
        if wait:
            return self.load_explanations(datasets)
        else:
            return jobs

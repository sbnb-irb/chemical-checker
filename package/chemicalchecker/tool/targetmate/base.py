"""


- TargetMate Stacked Classifier.

XXX

- TargetMate Ensemble Classifier.

An ensemble-based classifier based on CC signatures of different types.
A base classifier is specified, and predictions are made for each dataset
individually.

A meta-prediction is then provided based on individual
predictions, together with a measure of confidence for each prediction.
In the predictions, known data is provided as 1/0 predictions. The rest of
probabilities are clipped between 0.001 and 0.999.

- TargetMate Stacked Regressor.

XXX

- TargetMate Ensemble Regressor.

XXX

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
from sklearn import metrics
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import percentileofscore

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged
from chemicalchecker.util import Config

from .utils import metrics
from .utils import chemistry
from .universes import Universe
from .utils import plots
from .utils import HPCUtils


@logged
class TargetMateSetup(HPCUtils):
    """Set up the base TargetMate class"""

    def __init__(self,
                 models_path,
                 tmp_path = None,
                 cc_root = None,
                 n_jobs = None,
                 applicability = True,
                 **kwargs):
        """Basic setup of the TargetMate.

        Args:
            models_path(str): Directory where models will be stored.
            tmp_path(str): Directory where temporary data will be stored
                (relevant at predict time) (default=None)
            cc_root(str): CC root folder (default=None)
            n_jobs(int): Number of CPUs to use, all by default (default=None)
            cv(int): Number of cv folds (default=5)
            applicability(bool): Perform applicability domain calculation (default=True)
        """
        HPCUtils.__init__(self, **kwargs)
        # Jobs
        if not n_jobs:
            self.n_jobs = self.cpu_count()
        else:
            self.n_jobs = n_jobs
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(self.models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)
        self.bases_models_path, self.signatures_models_path = directory_tree(self.models_path)
        # Temporary path
        if not tmp_path:
            self.tmp_path = os.path.join(
                Config().PATH.CC_TMP, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        self.bases_tmp_path, self.signatures_tmp_path = directory_tree(self.tmp_path)
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Do applicability
        self.applicability = applicability
        # Others
        self._is_fitted  = False
        self._is_trained = False
        self.pipes       = []

    # Chemistry functions
    @staticmethod
    def read_smiles(smi, standardize):
        return chemistry.read_smiles(smi, standardize)

    # Other functions
    @staticmethod
    def directory_tree(root):
        bases_path = os.path.join(root, "bases")
        if not os.path.exists(bases_path): os.mkdir(bases_path)
        signatures_path = os.path.join(root, "signatures")
        if not os.path.exists(signatures_path): os.mkdir(signatures_path)
        return bases_path, signatures_path

    @staticmethod
    def avg_and_std(values, weights=None):
        """Return the (weighted) average and standard deviation.

        Args:
            values(list or array): 1-d list or array of values
            weights(list or array): By default, no weightening is applied
        """
        if weights is None:
            weights = np.ones(len(values))
        average = np.average(values, weights=weights)
        variance = np.average((values - average)**2, weights=weights)
        return (average, math.sqrt(variance))

    # Loading functions
    @staticmethod
    def load(models_path):
        """Load previously stored TargetMate instance."""
        with open(os.path.join(models_path, "/TargetMate.pkl", "r")) as f:
            return pickle.load(f)

    def load_performances(self):
        """Load performance data"""
        with open(os.path.join(self.models_path, "perfs.json"), "r") as f:
            return json.load(f)

    def load_ad_data(self):
        """Load applicability domain data"""
        with open(os.path.join(self.models_path, "ad_data.pkl"), "r") as f:
            return pickle.load(f)

    def load_data(self, data, standardize, use_checkpoints):
        if not use_checkpoints:
            # Cleaning models directory
            self.__log.debug("Cleaning previous checkpoints")
            shutil.rmtree(self.models_path)
            os.mkdir(self.models_path)
        # Read data
        self.__log.info("Reading data")
        # Read data if it is a file
        if type(data) == str:
            self.__log.info("Reading file %s", data)
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter="\t"):
                    data += [[r[0]] + r[1:]]
        # Get only valid SMILES strings
        self.__log.info(
            "Parsing SMILES strings, keeping only valid ones for training.")
        data_ = []
        for i, d in enumerate(data):
            m = self.read_smiles(d[1], standardize)
            if not m:
                continue
            # data is always of [(initial index, activity, ..., smiles,
            # inchikey)]
            data_ += [[i, d[0]] + [m[1], m[0]]]
        data = data_
        return data

    def load_base_model(self, destination_dir, append_pipe=False):
        """Load a base model"""
        mod = joblib.load(destination_dir)
        if append_pipe:
            self.pipes += [pickle.load(open(destination_dir+".pipe", "rb"))]
        return mod

    # Saving functions
    def save(self):
        """Save TargetMate instance"""
        # we avoid saving signature instances
        self.sign_predict_fn = None
        with open(self.models_path + "/TargetMate.pkl", "wb") as f:
            pickle.dump(self, f)

    def save_performances(self, perfs):
        with open(self.models_path + "/perfs.json", "w") as f:
            json.dump(perfs, f)

    def save_data(self, data):
        self.__log.debug("Saving training data (only evidence)")
        with open(self.models_path + "/trained_data.pkl", "wb") as f:
            pickle.dump(data, f)


@logged
class Fingerprinter(TargetMateSetup):
    """Set up a Fingerprinter. This is usually used as a baseline featurizer to compare with CC signatures."""

    def __init__(self, **kwargs):
        # Inherit TargetMateSetup
        TargetMateSetup.__init__(**kwargs)
        # Featurizer
        self.featurizer = chemistry.morgan_matrix
        self.dataset = "FP"

    def featurize(self, smiles, **kwargs):
        """Calculate fingerprints"""
        destination_dir = os.path.join(self.models_path, self.dataset)
        V = self.featurizer(smiles)
        with h5.File(destination_dir, "wb") as hf:
            hf.create_dataset("V", data = V.astype(np.int8))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))
    
    def read_fingerprint(self, idxs=None, fp_file=None, is_prd=False):
        """Read a signature from an HDF5 file"""
        # Identify HDF5 file
        if not fp_file:
            if is_prd:
                h5file = os.path.join(self.tmp_path, self.dataset)
            else:
                h5file = os.path.join(self.models_path, self.dataset)
        else:
            h5file = os.path.join(fp_file)
        # Read the file
        with h5py.File(h5file, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V


@logged
class Signaturizer(TargetMateSetup):
    """Set up a Signaturizer"""

    def __init__(self,
                 datasets=None,
                 sign_predict_fn=None,
                 **kwargs):
        """Set up a Signaturizer
        
        Args:
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign predictor are
                used.
            sign_predict_fn(dict): pre-loaded predict_fn, keys are dataset
                codes, values are tuples of (sign, predict_fn)
        """
        # Inherit TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)
        # Datasets
        if not datasets:
            # self.datasets = list(self.cc.datasets)
            self.datasets = ["%s%s.001" % (x, y)
                             for x in "ABCDE" for y in "12345"]
        else:
            self.datasets = datasets
        # preloaded neural networks
        if sign_predict_fn is None:
            self.sign_predict_fn = dict()
            for ds in self.datasets:
                self.__log.debug("Loading sign predictor for %s" % ds)
                s3 = self.cc.get_signature("sign3", "full", ds)
                self.sign_predict_fn[ds] = (s3, s3.get_predict_fn())
        else:
            self.sign_predict_fn = sign_predict_fn

    def get_destination_dir(self, dataset, is_prd):
        if is_prd:
            return os.path.join(self.signatures_tmp_path, dataset)
        else:
            return os.path.join(self.signatures_models_path, dataset)

    # Calculate signatures
    def signaturize_local(self, smiles, use_checkpoints, is_prd):
        self.__log.info("Calculating sign for every molecule.")
        for dataset in self.datasets:
            destination_dir = get_destination_dir(dataset, is_prd)
            if os.path.exists(destination_dir) and use_checkpoints:
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3, predict_fn = self.sign_predict_fn[dataset]
                s3.predict_from_smiles(smiles,
                                       destination_dir, predict_fn=predict_fn,
                                       use_novelty_model=False)

    def signaturize_hpc(self, smiles, chunk_size, use_checkpoints, is_prd):
        self.__log.info("Calculating sign for every molecule.")
        jobs  = []
        for dataset in self.datasets:
            destination_dir = os.path.join(self.models_path, dataset)
            if os.path.exists(destination_dir) and use_checkpoints:
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3, predict_fn = self.sign_predict_fn[dataset]
                job = s3.func_hpc("predict_from_smiles", smiles,
                            destination_dir, chunk_size, None, False, wait = False)
                jobs += [job]
        self.waiter(jobs)

    def signaturize(self, data, hpc, chunk_size=1000, use_checkpoints=False):
        if hpc:
            self.signaturize_hpc(data, chunk_size=chunk_size, use_checkpoints=use_checkpoints)
        else:
            self.signaturize_local(data, use_checkpoints=use_checkpoints)
     
    # Signature readers
    def read_signature(self, dataset, idxs=None, sign_folder=None, is_prd=False):
        """Read a signature from an HDF5 file"""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset, is_prd)
        else:
            destination_dir = os.path.join(sign_folder, dataset)
        with h5py.File(destination_dir, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V

    def read_signatures_ensemble(self, datasets=None, **kwargs):
        """Return signatures as an ensemble"""
        if not datasets: datasets = self.datasets
        for ds in datasets:
            yield self.read_signature(ds, **kwargs)

    def read_signatures_stacked(self, datasets=None, **kwargs):
        """Return signatures in a stacked form"""
        if not datasets: datasets = self.datasets
        V = []
        for ds in datasets:
            V += [self.read_signature(ds, **kwargs)]
        return np.hstack(V)


class ApplicabilityDomain(TargetMateSetup):
    """Applicability domain functionalities, inspired by conformal prediction methods"""

    def __init__(self,
                 k=5,
                 min_sim=0.25,
                 **kwargs):
        """Applicability domain parameters.
        
        Args:
            k(int): Number of molecules to look across when doing the
                applicability domain (default=5).
            min_sim(float): Minimum Morgan Tanimoto similarity to consider in
                the applicability domain determination (default=0.25).
        """
        # Inherit from TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)       
        # K-neighbors to search during the applicability domain calculation
        self.k = k
        # Minimal chemical similarity to consider
        self.min_sim = min_sim

    def fingerprint_arena(self, smiles, use_checkpoints=False, is_prd=False):
        if is_prd:
            fps_file = os.path.join(self.tmp_path, "arena.fps")
        else:
            fps_file = os.path.join(self.models_path, "arena.fps")
        if not use_checkpoints or not os.path.exists(fps_file):
            self.__log.debug("Writing Fingerprints")
            arena = chemistry.morgan_arena(smiles, fps_file)
        else:
            arena = chemistry.load_morgan_arena(fps_file)
        return arena

    @staticmethod
    def calculate_bias(yt, yp):
        """Calculate the bias of the QSAR model"""
        return np.array([np.abs(t - p) for t, p in zip(yt, yp)])

    def knearest_search(self, query_smi, target_fps, N):
        """Nearest neighbors search using chemical fingerprints"""
        k = np.min([self.k, len(target_fps.fps[0]) - 1])
        neighs = []
        for smi in query_smi:
            results = target_fps.similarity(
                smi, self.min_sim, n_workers=self.n_jobs)
            neighs += [results[:k]]
        if N is None:
            N = len(query_smi)
        sims = np.zeros((N, k), dtype=np.float32)
        idxs = np.zeros((N, k), dtype=np.int)
        for q_idx, hits in enumerate(neighs):
            sims_ = []
            idxs_ = []
            for h in hits:
                sims_ += [h[1]]
                idxs_ += [int(h[0])]
            sims[q_idx, :len(sims_)] = sims_
            idxs[q_idx, :len(idxs_)] = idxs_
        return sims, idxs

    def calculate_weights(self, query_smi, target_fps, stds, bias, N=None):
        """Calculate weights using adaptation of Aniceto et al 2016."""
        self.__log.debug("Finding nearest neighbors")
        sims, idxs = self.knearest_search(query_smi, target_fps, N)
        self.__log.debug("Calculating weights from std and bias")
        weights = []
        for i, idxs_ in enumerate(idxs):
            ws = sims[i] / (stds[idxs_] * bias[idxs_])
            weights += [np.max(ws)]
        return np.array(weights)


@logged
class TargetMate(TargetMateSetup, Signaturizer):

    def __init__(self, **kwargs):
        """XXX"""
        TargetMateSetup.__init__(self, **kwargs)
        Signaturizer.__init__(self, **kwargs)




@logged
class TargetMateClassifierSetup(TargetMateSetup):
    """Set up a TargetMate classifier. It can sample negative from a universe of molecules (e.g. ChEMBL)"""

    def __init__(self,
                 base_mod="logistic_regression",
                 cv=5,
                 min_class_size=10,
                 inactives_per_active=100,
                 metric="bedroc",
                 universe_path=None,
                 naive_sampling=False,
                 **kwargs):
        """Set up a TargetMate classifier

        Args:
            base_mod(clf): Classifier instance, containing fit and
                predict_proba methods (default="logistic_regression")
                The following strings are accepted: "logistic_regression",
                "random_forest", "naive_bayes" and "tpot"
                By default, sklearn LogisticRegressionCV is used.
            cv(int): Number of cv folds. The default cv generator used is
                Stratified K-Folds (default=5).
            min_class_size(int): Minimum class size acceptable to train the
                classifier (default=10).
            inactives_per_active(int): Number of inactive to sample for each active.
                If None, only experimental actives and inactives are considered (default=100).
            metric(str): Metric to use in the meta-prediction (bedroc, auroc or aupr)
                (default="bedroc").
            universe_path(str): Path to the universe. If not specified, the default one is used (default=None).
            naive_sampling(bool): Sample naively (randomly), without using the OneClassSVM (default=False).
        """
        # Inherit from TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)
        # Set the base classifier
        if type(base_mod) == str:
            if base_mod == "logistic_regression":
                from sklearn.linear_model import LogisticRegressionCV
                self.base_mod = LogisticRegressionCV(
                    cv=3, class_weight="balanced", max_iter=1000,
                    n_jobs=self.n_jobs)
            if base_mod == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.base_mod = RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", n_jobs=self.n_jobs)
            if base_mod == "naive_bayes":
                from sklearn.naive_bayes import GaussianNB
                self.base_mod = Pipeline(
                    [('feature_selection', VarianceThreshold()),
                     ('classify', GaussianNB())])
            if base_mod == "tpot":
                from tpot import TPOTClassifier
                from models import tpotconfigs
                self.base_mod = TPOTClassifier(
                    config_dict=tpotconfigs.minimal,
                    generations=10, population_size=30,
                    cv=3, scoring="balanced_accuracy",
                    verbosity=2, n_jobs=self.n_jobs,
                    max_time_mins=5, max_eval_time_mins=0.5,
                    random_state=42,
                    early_stop=3,
                    disable_update_check=True
                )
                self._is_tpot = True
            else:
                self._is_tpot = False
        else:
            self.base_mod = base_mod
        # Minimum size of the minority class
        self.min_class_size = min_class_size
        # Inactives per active
        self.inactives_per_active = inactives_per_active
        # Metric to use
        self.metric = metric
        # Load universe
        self.universe = Universe.load_universe(universe_path)
        # naive_sampling
        self.naive_sampling = naive_sampling

    @staticmethod
    def _reassemble_activity_sets(act, inact, putinact):
        data = []
        for x in list(act):
            data += [(x[1], 1, x[0], x[-1])]
        for x in list(inact):
            data += [(x[1], -1, x[0], x[-1])]
        n = np.max([x[0] for x in data]) + 1
        for i, x in enumerate(list(putinact)):
            data += [(i + n, 0, x[0], x[-1])]
        return data

    @staticmethod
    def performances(yt, yp):
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
        perfs["auroc"] = metrics.roc_score(yt, yp)
        perfs["aupr"] = metrics.pr_score(yt, yp)
        perfs["bedroc"] = metrics.bedroc_score(yt, yp)
        return perfs

    def prepare_data(self, data, standardize, use_checkpoints):
        # Read data
        data = self.load_data(data, standardize, use_checkpoints)
        # Convert activity data to integer
        data = [[d[0]] + [int(d[1])] + list(d[2:]) for d in data]
        # Save training data
        self.save_data(data)
        # Sample inactives, if necessariy
        if self.inactives_per_active:
            self.__log.info("Sampling putative inactives")
        actives = set([(d[-2], d[0], d[-1]) for d in data if d[1] == 1])
        inactives = set([(d[-2], d[0], d[-1]) for d in data if d[1] == -1])
        act, inact, putinact = self.universe.predict(actives, inactives,
                                                     inactives_per_active=self.inactives_per_active,
                                                     min_actives=self.min_class_size,
                                                     naive=self.naive_sampling)
        self.__log.info("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        self.__log.debug("Assembling and shuffling")
        data = self._reassemble_activity_sets(act, inact, putinact)
        random.shuffle(data)
        return data

    def prepare_for_ml(self, data):
        self.__log.debug("Prepare for machine learning")
        y = np.array([d[1] for d in data])
        # Consider putative inactives as inactives (i.e. set -1 to 0)
        self.__log.debug(
            "Considering putative inactives as inactives for training")
        y[y <= 0] = 0
        molecules = np.array([(d[-2], d[-1]) for d in data])
        smiles = np.array([m[0] for m in molecules])
        # Check that there are enough molecules for training.
        self.ny = np.sum(y)
        if self.ny < self.min_class_size or (len(y) - self.ny) < self.min_class_size:
            self.__log.warning(
                "Not enough valid molecules in the minority class..." +
                "Just keeping training data")
            self._is_fitted = True
            self.save()
            return
        self.__log.info("Actives %d / Merged inactives %d" % (self.ny, len(y) - self.ny))
        # Results
        results ={
            "y": y,
            "molecules": molecules,
            "smiles": smiles
        }
        return results

    def fitter(self, X, y, destination_dir=None, is_cv=False, pipe=None, n_jobs=None):
        """Fit a model.
        
        Args:
            X(array): Signatures matrix.
            y(array): Labels vector.
            destination_dir(str): File where to store the model.
                If not specified, the model is returned in memory (default=None).
            is_cv(bool): Is the fit part of a cross-validation regime.
                This affect the usage of pre-identified pipelines, for example, with TPOT (default=False).
            pipe(ML pipeline): A machine-learning pipeline.
            n_jobs(int): If jobs are specified, the number of CPUs per model are overwritten.
                This is relevant when sending jobs to the cluster (default=None).
        """
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        if not is_cv:  
            mod = clone(self.base_mod)
            if n_jobs: mod.n_jobs = n_jobs
            mod.fit(X[shuff], y[shuff])
            if self._is_tpot:
                self.pipes += [mod.fitted_pipeline_]
                mod = self.pipes[-1]
                if n_jobs: mod.n_jobs = n_jobs
                mod = mod.fit(X[shuff], y[shuff])
            else:
                self.pipes += [None]
        else:
            if not pipe:
                mod = clone(self.base_mod)
            else:
                mod = pipe
            if n_jobs: mod.n_jobs = n_jobs
            mod.fit(X, y)
        mod.n_jobs = self.n_jobs
        if destination_dir:
            joblib.dump(mod , destination_dir)
            pickle.dump(pipe, open(destination_dir+".pipe", "wb"))
        else:
            return mod

    def predictor(self, mod, X):
        """Make predictions.
        
        Args:
            mod(model): Predictive model.
            X(array): Signatures.
        """
        return [p[1] for p in mod.predict_proba(X_train)]


@logged
class TargetMateRegressionSetup(TargetMateSetup):
    """Set up a TargetMate classifier"""

    def __init__(self, base_mod = "", metric = ""):
        # Configure the base regressor
        self.base_mod = base_mod
        # Metric to use
        self.metric = metric

    @staticmethod
    def performances(yt, yp):
        perfs = {}
        return perfs


@logged
class TargetMateStackedClassifier(TargetMateClassifierSetup):
    pass


@logged
class TargetMateEnsembleClassifier(TargetMateClassifierSetup, Signaturizer):
    """TargetMate ensemble classifier"""

    def __init__(self, **kwargs):
        """TargetMate ensemble classifier"""
        
        TargetMateClassifierSetup.__init__(**kwargs)
        Signaturizer.__init__(**kwargs)
        self.ensemble = []

    def fit_ensemble_local(self, y):
        self.__log.info("Local fit of individual models (full data)")
        for i, X in enumerate(self.read_signatures_ensemble(self, datasets=self.datasets)):
            dest = os.path.join(self.bases_models_path, self.datasets[i])
            self.ensemble += [(self.datasets[i], dest)]
            self.fitter(X, y, destination_dir=dest, is_cv=False, pipe=None, n_jobs=None)

    def fit_ensemble_hpc(self, y, n_jobs=16):
        self.__log.info("HPC fit of individual models (full data)")
        jobs  = []
        for i, X in enumerate(self.read_signatures_ensemble(self, datasets=self.datasets)):
            dest = os.path.join(self.bases_tmp_path, self.datasets[i])
            self.ensemble += [(self.datasets[i], dest)]
            jobs  += [self.func_hpc("fitter", X, y, dest, False, None, cpu=n_jobs)]
        self.waiter(jobs)

    def fit_ensemble(self, y):
        if not self.hpc:
            fit_ensemble_local(y)
        else:
            fit_ensemble_hpc(y)

    def ensemble_iter(self):
        for ds, dest in self.ensemble:
            yield self.load_base_model(dest)

    def cross_validation_local(self, y):
        # Initialize cross-validation generator
        skf = StratifiedKFold(n_splits=np.min(
            [self.cv, self.ny]), shuffle=True, random_state=42)
        # Do the individual predictors
        self.__log.info("Training individual predictors with cross-validation")
        yps_train = collections.defaultdict(list)
        yps_test  = collections.defaultdict(list)
        yts_train = []
        yts_test  = []
        smi_test  = []
        for train_idx, test_idx in skf.split(smiles, y):
            self.__log.debug("CV fold")
            for i, dataset in enumerate(self.datasets):
                X_train = self.read_signature(dataset, train_idx, is_prd = False)
                y_train = y[train_idx]
                mod = self.fitter(X_train, y_train, is_cv = True, self.pipes[i])
                # Make predictions on train set itself
                yps_train[dataset] += self.predictor(X_train)
                # Make predictions on test set itself
                X_test = self.read_signature(dataset, test_idx, is_prd = False)
                y_test = y[test_idx]
                yps_test[dataset] += self.predictor(X_test)
            yts_train += list(y_train)
            yts_test  += list(y_test)
            smi_test  += list(smiles[test_idx])
        results = {
            "yps_train": yps_train,
            "yps_test": yps_test,
            "yts_train": yts_train,
            "yts_test": yts_test,
            "smi_test": smi_test
        }
        return results

    def metapredict(self, yp_dict, perfs, dataset_universe=None):
        """Do meta-prediction based on dataset-specific predictions.
        Weights are given according to the performance of the individual
        predictors.
        Standard deviation across predictions is kept to estimate
        applicability domain.
        """
        if dataset_universe is None:
            dataset_universe = set(self.datasets)
        M = []
        w = []
        for dataset in self.datasets:
            if dataset not in dataset_universe:
                continue
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

    def all_performances(self, cv_results):
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
            ptest = self.performances(yts_test, yps_test[dataset])
            perfs[dataset] = {"perf_train": ptrain, "perf_test": ptest}
        # Meta-predictor on train and test data
        self.__log.info("Meta-predictions on train and test data")
        self.__log.debug("Assembling for train set")
        mps_train, std_train = self.metapredict(yps_train, perfs)
        self.__log.debug("Assembling for test set")
        mps_test, std_test = self.metapredict(yps_test, perfs)
        # Assess meta-predictor performance
        self.__log.debug("Assessing meta-predictor performance")
        ptrain = self.performances(yts_train, mps_train)
        ptest = self.performances(yts_test, mps_test)
        perfs["MetaPred"] = {"perf_train": ptrain, "perf_test": ptest}
        results = {
            "perfs": perfs,
            "mps": ""
        }
        return results

    def plot(self):
        perfs = self.load_performances()
        ad_data = self.load_ad_data()
        plots.ensemble_classifier_grid(perfs, ad_data)

    def fit(self, use_checkpoints, hpc):
        # Read data
        data = self.prepare_data(use_checkpoints)
        # Prepare data for machine learning
        data_ml = self.prepare_for_ml(data)
        if not data_ml: return
        # Signaturize
        self.signaturize(data, hpc)
        # Fit the global classifier
        clf_ensemble = self.fit_ensemble()
        # Cross-validation
        cv_results = self.cross_validation()
        # Get performances
        ap_results = self.all_performances(cv_results)
        # 


    def predict(self):
        pass


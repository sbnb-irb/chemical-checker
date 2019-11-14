"""Set up TargetMate"""

import os
import shutil
import uuid
import csv
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold

from chemicalchecker.util import logged
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import Config

from .utils import HPCUtils
from .utils import chemistry
from .utils import conformal
from .io import InputData
from .universes import Universe
from .models.vanillaconfigs import VanillaClassifierConfigs


@logged
class TargetMateSetup(HPCUtils):
    """Set up the base TargetMate class"""

    def __init__(self,
                 models_path,
                 tmp_path = None,
                 cc_root = None,
                 overwrite = True,
                 n_jobs = None,
                 n_jobs_hpc = 1,
                 standardize = True,
                 cv_folds = 5,
                 conformity = True,
                 hpc = False,
                 do_init = True,
                 train_timeout = 3600,
                 **kwargs):
        """Basic setup of the TargetMate.

        Args:
            models_path(str): Directory where models will be stored.
            tmp_path(str): Directory where temporary data will be stored
                (relevant at predict time) (default=None)
            cc_root(str): CC root folder (default=None)
            overwrite(bool): Clean models_path directory (default=True)
            n_jobs(int): Number of CPUs to use, all by default (default=None)
            n_jobs(hpc): Number of CPUs to use in HPC (default=1)
            standardize(bool): Standardize small molecule structures (default=True)
            cv_folds(int): Number of cross-validation folds (default=5)
            conformity(bool): Do cross-conformal prediction (default=True)
            hpc(bool): Use HPC (default=False)
            train_timeout(int): Maximum time in seconds for training a classifier; applies to autosklearn (default=3600).
        """
        if not do_init:
            return
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
            os.makedirs(self.models_path, exist_ok = True)
        else:
            if overwrite:
                # Cleaning models directory
                self.__log.debug("Cleaning %s" % self.models_path)
                shutil.rmtree(self.models_path, ignore_errors=True)
                os.makedirs(self.models_path, exist_ok = True)
        self.bases_models_path, self.signatures_models_path, self.predictions_models_path = self.directory_tree(self.models_path)
        self._bases_models_path, self._signatures_models_path, self._predictions_models_path = self.bases_models_path, self.signatures_models_path, self.predictions_models_path 
        # Temporary path
        if not tmp_path:
            self.tmp_path = os.path.join(
                Config().PATH.CC_TMP, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.bases_tmp_path, self.signatures_tmp_path, self.predictions_tmp_path = self.directory_tree(self.tmp_path)
        self._bases_tmp_path, self._signatures_tmp_path, self._predictions_tmp_path = self.bases_tmp_path, self.signatures_tmp_path, self.predictions_tmp_path
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Standardize
        self.standardize = standardize
        # Do conformal modeling
        self.conformity = conformity
        # CV folds
        self.cv_folds = cv_folds
        # Use HPC
        self.n_jobs_hpc = n_jobs_hpc
        self.hpc = hpc
        # Timeout
        self.train_timeout = train_timeout
        # Others
        self._is_fitted  = False
        self._is_trained = False
        self.is_tmp      = False
        # Log path information
        self.__log.info("MODELS PATH: %s" % self.models_path)
        self.__log.info("TMP PATH: %s" % self.tmp_path)

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
        predictions_path = os.path.join(root, "predictions")
        if not os.path.exists(predictions_path): os.mkdir(predictions_path)
        return bases_path, signatures_path, predictions_path

    def reset_path_bases(self, is_tmp=True):
        if is_tmp:
            self.bases_tmp_path = self._bases_tmp_path
        else:
            self.bases_models_path = self._bases_models_path

    def repath_bases_by_fold(self, fold_number, is_tmp=True, reset=True):
        """Redefine path of a TargetMate instance. Used by the Validation class."""
        if reset:
            self.reset_path_bases(is_tmp=is_tmp)
        if is_tmp:
            self.bases_tmp_path = os.path.join(self.bases_tmp_path, "%02d" % fold_number)
            if not os.path.exists(self.bases_tmp_path): os.mkdir(self.bases_tmp_path)
        else:
            self.bases_models_path = os.path.join(self.bases_models_path, "%02d" % fold_number)
            if not os.path.exists(self.bases_models_path): os.mkdir(self.bases_models_path)

    def reset_path_predictions(self, is_tmp=True):
        """Reset predictions path"""
        if is_tmp:
            self.predictions_tmp_path = self._predictions_tmp_path
        else:
            self.predictions_models_path = self._predictions_models_path

    def repath_predictions_by_fold(self, fold_number, is_tmp=True, reset=True):
        """Redefine path of a TargetMate instance. Used by the Validation class."""
        if reset:
            self.reset_path_predictions(is_tmp=is_tmp)
        if is_tmp:
            self.predictions_tmp_path = os.path.join(self.predictions_tmp_path, "%02d" % fold_number)
            if not os.path.exists(self.predictions_tmp_path): os.mkdir(self.predictions_tmp_path)
        else:
            self.predictions_models_path = os.path.join(self.predictions_models_path, "%02d" % fold_number)
            if not os.path.exists(self.predictions_models_path): os.mkdir(self.predictions_models_path)

    def repath_predictions_by_set(self, is_train, is_tmp=True, reset=True):
        """Redefine path of a TargetMate instance. Used by the Validation class."""
        if reset:
            self.reset_path_predictions(is_tmp=is_tmp)
        if is_train:
            s = "train"
        else:
            s = "test"
        if is_tmp:
            self.predictions_tmp_path = os.path.join(self.predictions_tmp_path, s)
            if not os.path.exists(self.predictions_tmp_path): os.mkdir(self.predictions_tmp_path)
        else:
            self.predictions_models_path = os.path.join(self.predictions_models_path, s)
            if not os.path.exists(self.predictions_models_path): os.mkdir(self.predictions_models_path)
    
    def repath_predictions_by_fold_and_set(self, fold_number, is_train, is_tmp=True, reset=True):
        self.repath_predictions_by_fold(fold_number=fold_number, is_tmp=is_tmp, reset=reset)
        self.repath_predictions_by_set(is_train=is_train, is_tmp=is_tmp, reset=False)

    # Read input data
    def read_data(self, data, standardize=None):
        if not standardize:
            standardize = self.standardize
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
            # data is always of [(initial index, activity, smiles,
            # inchikey)]
            data_ += [[i, float(d[0])] + [m[1], m[0]]]
        data = data_
        return InputData(data)

    # Loading functions
    @staticmethod
    def load(models_path):
        """Load previously stored TargetMate instance."""
        with open(os.path.join(models_path, "/TargetMate.pkl", "r")) as f:
            return pickle.load(f)
 
    def load_base_model(self, destination_dir, append_pipe=False):
        """Load a base model"""
        mod = joblib.load(destination_dir)
        if append_pipe:
            self.pipes += [pickle.load(open(destination_dir+".pipe", "rb"))]
        return mod

    def load_data(self):
        self.__log.debug("Loading training data (only evidence)")
        fn = os.path.join(self.models_path, "trained_data.pkl")
        if not os.path.exists(fn): return
        with open(fn, "rb") as f:
            return pickle.load(f)

    # Saving functions
    def save(self):
        """Save TargetMate instance"""
        # we avoid saving signature instances
        self.sign_predict_fn = None
        with open(self.models_path + "/TargetMate.pkl", "wb") as f:
            pickle.dump(self, f)

    def save_data(self, data):
        self.__log.debug("Saving training data (only evidence)")
        with open(self.models_path + "/trained_data.pkl", "wb") as f:
            pickle.dump(data, f)

    # Wipe
    def wipe(self):
        """Delete temporary data"""
        self.__log.debug("Removing %s" % self.tmp_path)
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        for job_path in self.job_paths:
            if os.path.exists(job_path):
                self.__log.debug("Removing %s" % job_path)
                shutil.rmtree(job_path, ignore_errors=True)


@logged
class TargetMateClassifierSetup(TargetMateSetup):
    """Set up a TargetMate classifier. It can sample negatives from a universe of molecules (e.g. ChEMBL)"""

    def __init__(self,
                 algo=None,
                 model_config="autosklearn",
                 weight_algo="naive_bayes",
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
            algo(str): Base algorithm to use (see /model configuration files) (default=random_forest).
            model_config(str): Model configurations for the base classifier (default=vanilla).
            weight_algo(str): Model used to weigh the contribution of an individual classifier.
                Should be fast. For the moment, only vanilla classifiers are accepted (default=naive_bayes).
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
            self.algo = VanillaClassifierConfigs(self.algo, n_jobs=self.n_jobs)
        if self.model_config == "tpot":
            from .models.tpotconfigs import TPOTClassifierConfigs
            self.algo = TPOTClassifierConfigs(self.algo, n_jobs=self.n_jobs)
        if self.model_config == "autosklearn":
            from .models.autosklearnconfigs import AutoSklearnClassifierConfigs
            self.algo = AutoSklearnClassifierConfigs(n_jobs=self.n_jobs, tmp_path=self.tmp_path, train_timeout=self.train_timeout)
        # Weight algo
        self.weight_algo = VanillaClassifierConfigs(weight_algo, n_jobs=self.n_jobs)
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
        self.cross_conformal_func = conformal.get_cross_conformal_classifier

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

    def kfolder(self):
        """Cross-validation splits strategy"""
        return StratifiedKFold(n_splits=int(np.min([self.cv_folds, self.ny])),
                               shuffle=True, random_state=42)


@logged
class TargetMateRegressorSetup(TargetMateSetup):
    """Set up a TargetMate classifier"""

    pass


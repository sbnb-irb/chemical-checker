"""Set up TargetMate"""

import os
import shutil
import uuid
import pickle
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, KFold

from chemicalchecker.util import logged
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import Config

from .utils import HPCUtils
from .utils import conformal
from .utils.log import set_logging
from .io import read_data, reassemble_activity_sets
from .universes import Universe
from .models.vanillaconfigs import VanillaClassifierConfigs


@logged
class TargetMateSetup(HPCUtils):
    """Set up the base TargetMate class"""

    def __init__(self,
                 models_path,
                 tmp_path = None,
                 cc_root = None,
                 is_classic = False,
                 classic_dataset = "A1.001",
                 classic_cctype = "sign0",
                 prestacked_dataset = None,
                 overwrite = True,
                 n_jobs = None,
                 n_jobs_hpc = 8,
                 max_train_samples = 10000,
                 max_train_ensemble = 10,
                 train_sample_chance = 0.95,
                 standardize = False,
                 is_cv = False,
                 is_stratified = True,
                 n_splits = 3,
                 test_size_hyperopt = 0.2,
                 scaffold_split = False,
                 outofuniverse_split = False,
                 outofuniverse_datasets = ["A1.001"],
                 outofuniverse_cctype = "sign1",
                 conformity = True,
                 hpc = False,
                 do_init = True,
                 search_n_iter = 25,
                 train_timeout = 7200,
                 shuffle = False,
                 log = "INFO",
                 use_stacked_signature=False,
                 is_tmp_bases=True,
                 is_tmp_signatures=True,
                 is_tmp_predictions=True,
                 use_cc = True,
                 **kwargs):
        """Basic setup of the TargetMate.

        Args:
            models_path(str): Directory where models will be stored.
            tmp_path(str): Directory where temporary data will be stored
                (relevant at predict time) (default=None).
            cc_root(str): CC root folder (default=None).
            is_classic(bool): Use a classical chemical fingerprint, instead of CC signatures (default=False).
            classic_dataset(str): Dataset code for the classic fingerprint.
            classic_cctype(str): Signature for the classic dataset.
            prestacked_dataset(str): Prestacked dataset signature.
            overwrite(bool): Clean models_path directory (default=True).
            n_jobs(int): Number of CPUs to use, all by default (default=None).
            n_jobs_hpc(int): Number of CPUs to use in HPC (default=1).
            max_train_samples(int): Maximum number of training samples to use (default=10000).
            max_train_ensemble(int): Maximum size of an ensemble (important when many samples are available) (default=10).
            train_sample_chance(float): Chance of visiting a sample (default=0.95).
            standardize(bool): Standardize small molecule structures (default=True).
            is_cv(bool): In hyper-parameter optimization, do cross-validation (default=False).
            is_stratified(bool): In hyper-parameter optimization, do stratified split (default=True).
            n_splits(int): If hyper-parameter optimization is done, number of splits (default=3).
            test_size_hyperopt(int): If hyper-parameter optimization is done, size of the test (default=0.2).
            scaffold_split(bool): Model should be evaluated with scaffold splits (default=False).
            outofuniverse_split(bool): Model should be evaluated with out-of-universe splits (default=False).
            outofuniverse_datasets(list): Datasets to consider as part of the universe in the out-of-universe split.
            outofuniverse_cctype(str): Signature type of the datasets considered to be part of the out-of-universe split.
            conformity(bool): Do cross-conformal prediction (default=True)
            hpc(bool): Use HPC (default=False)
            search_n_iter(int): Number of iterations in a search for hyperparameters (default=25).
            train_timeout(int): Maximum time in seconds for training a classifier; applies to autosklearn (default=7200).
            use_cc(bool): Use pre-computed CC signatures.
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
        # Temporary path
        if not tmp_path:
            subpath = self.models_path.rstrip("/").split("/")[-1]
            self.tmp_path = os.path.join(
                Config().PATH.CC_TMP, "targetmate", subpath, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.join(os.path.abspath(tmp_path), str(uuid.uuid4()))

        self.is_tmp_bases = is_tmp_bases
        self.is_tmp_signatures = is_tmp_signatures
        self.is_tmp_predictions = is_tmp_predictions
        if not os.path.exists(self.tmp_path): os.makedirs(self.tmp_path, exist_ok = True)
        self.bases_tmp_path, self.signatures_tmp_path, self.predictions_tmp_path = self.directory_tree(self.tmp_path)
        self._bases_tmp_path, self._signatures_tmp_path, self._predictions_tmp_path = self.bases_tmp_path, self.signatures_tmp_path, self.predictions_tmp_path
        self.arrays_tmp_path = os.path.join(self.tmp_path, "arrays")
        os.makedirs(self.arrays_tmp_path, exist_ok = True)
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Use classical or CC fingerprint
        self.is_classic = is_classic
        self.classic_dataset = classic_dataset
        self.classic_cctype = classic_cctype
        # Stacked signature
        self.use_stacked_signature = use_stacked_signature
        self.prestacked_dataset = prestacked_dataset
        # Standardize
        self.standardize = standardize
        # Do conformal modeling
        self.conformity = conformity
        # Topping a classifier with a determined number of samples
        self.max_train_samples = max_train_samples
        self.max_train_ensemble = max_train_ensemble
        self.train_sample_chance = train_sample_chance
        # Do cross-validation
        self.overwrite = overwrite
        self.is_cv = is_cv
        # Stratified
        self.is_stratified = is_stratified
        # Number os splits
        self.n_splits = n_splits
        # Test size
        self.test_size_hyperopt = test_size_hyperopt
        # Scaffold splits
        self.scaffold_split = scaffold_split
        # Out-of-universe splits
        self.outofuniverse_split = outofuniverse_split
        if outofuniverse_datasets is None:
            self.outofuniverse_datasets = ["A1.001"]
        else:
            self.outofuniverse_datasets = outofuniverse_datasets
        self.outofuniverse_cctype = outofuniverse_cctype
        # Use HPC
        self.n_jobs_hpc = n_jobs_hpc
        self.hpc = hpc
        # Grid iterations
        self.search_n_iter = search_n_iter
        # Timeout
        self.train_timeout = train_timeout
        # Shuffle
        self.shuffle = shuffle
        # Logging
        self.log = log
        # set_logging(self.log)
        # Others
        self._is_fitted  = False
        self._is_trained = False
        # self.is_tmp      = False
        # Log path information
        self.__log.info("MODELS PATH: %s" % self.models_path)
        self.__log.info("TMP PATH: %s" % self.tmp_path)
        self.use_cc = use_cc
    # Directories functions
    @staticmethod
    def directory_tree(root):
        bases_path = os.path.join(root, "bases")
        if not os.path.exists(bases_path): os.mkdir(bases_path)
        signatures_path = os.path.join(root, "signatures")
        if not os.path.exists(signatures_path): os.mkdir(signatures_path)
        predictions_path = os.path.join(root, "predictions")
        if not os.path.exists(predictions_path): os.mkdir(predictions_path)
        return bases_path, signatures_path, predictions_path

    def create_models_path(self):
        if not os.path.exists(self.models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.makedirs(self.models_path, exist_ok = True)
        else:
            if self.overwrite:
                # Cleaning models directory
                self.__log.debug("Cleaning %s" % self.models_path)
                shutil.rmtree(self.models_path, ignore_errors=True)
                os.makedirs(self.models_path, exist_ok = True)
        self.bases_models_path, self.signatures_models_path, self.predictions_models_path = self.directory_tree(self.models_path)
        self._bases_models_path, self._signatures_models_path, self._predictions_models_path = self.bases_models_path, self.signatures_models_path, self.predictions_models_path

    def reset_path_bases(self):
        if self.is_tmp_bases:
            self.bases_tmp_path = self._bases_tmp_path
        else:
            self.bases_models_path = self._bases_models_path

    def repath_bases_by_fold(self, fold_number, is_tmp = True, reset=True, only_train = False):
        """Redefine path of a TargetMate instance. Used by the Validation class."""
        if reset:
            self.reset_path_bases()
        if not only_train:
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

    def repath_predictions_by_fold_and_set(self, fold_number, is_train, is_tmp=True, reset=True, only_train = False):
        if not only_train:
            self.repath_predictions_by_fold(fold_number=fold_number, is_tmp=is_tmp, reset=reset)
            self.repath_predictions_by_set(is_train=is_train, is_tmp=is_tmp, reset=False)
        else:
            self.repath_predictions_by_set(is_train=is_train, is_tmp=is_tmp, reset=True)

    # Read input data
    def read_data(self, data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey, standardize=None, valid_inchikeys=None):
        if not standardize:
            standardize = self.standardize
        # Read data
        self.__log.info("Reading data, parsing molecules")
        return read_data(data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, standardize, use_inchikey, valid_inchikeys=valid_inchikeys)

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

    def compress_models(self):
        """Store model in compressed format for persistance"""
        mod_dir = self.bases_models_path
        for m in os.listdir(mod_dir):
            fn = os.path.join(mod_dir, m)
            mod = joblib.load(fn)
            joblib.dump(mod, fn + ".z")
            os.remove(fn)

@logged
class TargetMateClassifierSetup(TargetMateSetup):
    """Set up a TargetMate classifier. It can sample negatives from a universe of molecules (e.g. ChEMBL)"""

    def __init__(self,
                 algo=None,
                 model_config="autosklearn",
                 weight_algo="naive_bayes",
                 ccp_folds=10,
                 min_class_size=10,
                 min_class_size_active=None, # Added by Paula: different number of active/inactive minimum value
                 min_class_size_inactive=None, # Added by Paula: different number of active/inactive minimum value
                 active_value=1,
                 inactive_value=None,
                 inactives_per_active=100,
                 metric="bacc",
                 universe_path=None,
                 naive_sampling=False,
                 biased_universe=0,
                 maximum_potential_actives = 5,
                 universe_random_state = None,
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
            min_class_size_active(int): Minimum active class size acceptable to train the
                classifier, if not stated, uses min_class_size (default=None).
            min_class_size_inactive(int): Minimum inactive class size acceptable to train the
                classifier, if not stated, uses min_class_size (default=None).
            active_value(int): When reading data, the activity value considered to be active (default=1).
            inactive_value(int): When reading data, the activity value considered to be inactive. If none specified,
                then any value different that active_value is considered to be inactive (default=None).
            inactives_per_active(int): Number of inactive to sample for each active.
                If None, only experimental actives and inactives are considered (default=100).
            metric(str): Metric to use to select the pipeline (default="auroc").
            universe_path(str): Path to the universe. If not specified, the default one is used (default=None).
            naive_sampling(bool): Sample naively (randomly), without using the OneClassSVM (default=False).
            biased_universe(float): Proportion of closer molecules to sample as putative inactives (default = 0).

        """
        # Inherit from TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)
        # Metric to use
        self.metric = metric
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
            self.algo = VanillaClassifierConfigs(self.algo,
                                                 n_jobs=n_jobs)
        if self.model_config == "grid":
            from .models.gridconfigs import GridClassifierConfigs
            self.algo = GridClassifierConfigs(self.algo,
                                              n_jobs=n_jobs,
                                              n_iter=self.search_n_iter)
        if self.model_config == "hyperopt":
            from .models.hyperoptconfigs import HyperoptClassifierConfigs
            self.algo = HyperoptClassifierConfigs(self.algo,
                                                  metric=self.metric,
                                                  n_jobs=n_jobs,
                                                  n_iter=self.search_n_iter,
                                                  timeout=self.train_timeout,
                                                  is_cv=self.is_cv,
                                                  is_stratified=self.is_stratified,
                                                  n_splits=self.n_splits,
                                                  test_size=self.test_size_hyperopt,
                                                  scaffold_split=self.scaffold_split)
        if self.model_config == "tpot":
            from .models.tpotconfigs import TPOTClassifierConfigs
            self.algo = TPOTClassifierConfigs(self.algo,
                                              n_jobs=n_jobs)
        if self.model_config == "autosklearn":
            from .models.autosklearnconfigs import AutoSklearnClassifierConfigs
            self.algo = AutoSklearnClassifierConfigs(n_jobs=n_jobs,
                                                     tmp_path=self.tmp_path,
                                                     train_timeout=self.train_timeout,
                                                     log=self.log)
        # Weight algo
        self.weight_algo = VanillaClassifierConfigs(weight_algo, n_jobs=self.n_jobs) # TO-DO: This is run locally for now.

        # Minimum size of the minority class
        if min_class_size_active is None and min_class_size_inactive is None: # Added by Paula: change number of actives/inactives per model
            self.min_class_size_active = min_class_size
            self.min_class_size_inactive = min_class_size
        elif min_class_size_active is not None and min_class_size_inactive is None:
            self.min_class_size_active = min_class_size_active
            self.min_class_size_inactive = min_class_size
        elif min_class_size_active is None and min_class_size_inactive is not None:
            self.min_class_size_active = min_class_size
            self.min_class_size_inactive = min_class_size_inactive
        else:
            self.min_class_size_active = min_class_size_active
            self.min_class_size_inactive = min_class_size_inactive

        # Active value
        self.active_value = active_value
        # Inactive value
        self.inactive_value = inactive_value
        # Inactives per active
        self.inactives_per_active = inactives_per_active
        # Load universe
        self.universe = Universe.load_universe(universe_path)
        # naive_sampling
        self.naive_sampling = naive_sampling
        # Others
        self.cross_conformal_func = conformal.get_cross_conformal_classifier
        self.biased_universe = biased_universe
        self.universe_random_state = universe_random_state
        self.maximum_potential_actives = maximum_potential_actives

    def _reassemble_activity_sets(self, act, inact, putinact, inchi=False):
        self.__log.info("Reassembling activities. Convention: 1 = Active, -1 = Inactive, 0 = Sampled")
        return reassemble_activity_sets(act, inact, putinact, inchi=inchi)

    def prepare_data(self, data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey):
        # Read data
        if self.use_cc:
            s = self.cc.signature(self.classic_dataset, self.classic_cctype)
            valid_inchikeys = s.keys
        else:
            valid_inchikeys = None
        data = self.read_data(data, smiles_idx=smiles_idx, inchi_idx=inchi_idx, inchikey_idx=inchikey_idx, activity_idx=activity_idx, srcid_idx=srcid_idx, use_inchikey=use_inchikey, valid_inchikeys=valid_inchikeys)
        self.ny = np.sum(data.activity == 1)
        if self.ny < self.min_class_size_active or (len(data.activity) - self.ny) < self.min_class_size_inactive: # Added by Paula: different number of actives or inactivess
            self.__log.warning("Not enough data (%d)" % self.ny)
            return None
        # Create file structure
        self.create_models_path()
        # Save training data
        self.save_data(data)
        # Sample inactives, if necessary
        actives   = set()
        inactives = set()
        for d in data:
            if d.activity == self.active_value:
                actives.update([(d.molecule, d.idx, d.inchikey)])
            else:
                if not self.inactive_value:
                    inactives.update([(d.molecule, d.idx, d.inchikey)])
                else:
                    if d.activity == self.inactive_value:
                        inactives.update([(d.molecule, d.idx, d.inchikey)])
        act, inact, putinact, self.putative_idx = self.universe.predict(actives, inactives,
                                                     inactives_per_active=self.inactives_per_active,
                                                     min_actives=self.min_class_size_active, # Added by Paula: change to specifically active class
                                                     naive=self.naive_sampling,
                                                     biased_universe=self.biased_universe,
                                                     maximum_potential_actives = self.maximum_potential_actives,
                                                     random_state= self.universe_random_state) # Added by Paula: sample proportion of universe closer to actives
        self.__log.info("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        print("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        self.__log.debug("Assembling")
        inchi = (smiles_idx is None) and (inchi_idx is not None)
        data = self._reassemble_activity_sets(act, inact, putinact, inchi)
        if self.shuffle:
            self.__log.debug("Shuffling")
            data.shuffle()
        return data

    def prepare_for_ml(self, data, predict=False):
        if data is None:
            return None
        """Prepare data for ML, i.e. convert to 1/0 and check that there are enough samples for training"""
        self.__log.debug("Prepare for machine learning (converting to 1/0")
        # Consider putative inactives as inactives (e.g. set -1 to 0)
        self.__log.debug("Considering putative inactives as inactives for training")
        data.activity[data.activity <= 0] = 0
        # Check that there are enough molecules for training.
        self.ny = np.sum(data.activity)
        if self.ny < self.min_class_size_active or (len(data.activity) - self.ny) < self.min_class_size_inactive: # Added by Paula: seperate minimums for actives and inactives
            self.__log.warning(
                "Not enough valid molecules in the minority class..." +
                "Just keeping training data")
            self._is_fitted = True
            #self.save()
            return None
        self.__log.info("Actives %d / Merged inactives %d" % (self.ny, len(data.activity) - self.ny))
        return data


@logged
class TargetMateRegressorSetup(TargetMateSetup):
    """Set up a TargetMate regressor"""

    pass



class ModelSetup(TargetMateClassifierSetup, TargetMateRegressorSetup):

    def __init__(self, is_classifier, **kwargs):
        if is_classifier:
            TargetMateClassifierSetup.__init__(self, **kwargs)
        else:
            TargetMateRegressorSetup.__init__(self, **kwargs)
        self.is_classifier = is_classifier

    def prepare_data(self, data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey):
        if self.is_classifier:
            return TargetMateClassifierSetup.prepare_data(self, data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey)
        else:
            return TargetMateRegressorSetup.prepare_data(self, data, smiles_idx, inchi_idx, inchikey_idx, activity_idx, srcid_idx, use_inchikey)

    def prepare_for_ml(self, data, predict=False):
        if self.is_classifier:
            return TargetMateClassifierSetup.prepare_for_ml(self, data, predict=predict)
        else:
            return TargetMateRegressorSetup.prepare_for_ml(self, data)

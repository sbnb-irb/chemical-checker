"""TargetMate Classifier.

    An ensemble-based classifier based on CC signatures of different types.
    A base classifier is specified, and predictions are made for each dataset
    individually. A meta-prediction is then provided based on individual predictions,
    together with a measure of confidence for each prediction.
    In the predictions, known data is provided as 1/0 predictions. The rest of probabilities are clipped between 0.001 and 0.999.
    In order to make results more interpretable, in the applicability domain we use chemical similarity for now.
    The basis for the applicability domain application can be found: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0182-y
    Obviously, CC signature similarities could be used in the future.
    The classifier is greatly inspired by PidginV3: https://pidginv3.readthedocs.io/en/latest/usage/index.html
"""
import os, shutil
import json
import h5py
import collections
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.base import clone
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from standardiser import standardise
from rdkit import Chem
import chemfp
import math
from scipy.stats import percentileofscore

@logged
class TargetMate:
    """TargetMate class"""

    def __init__(self, models_path, tmp_path = None, base_clf = "tpot", cv = 5, k = 5, min_sim = 0.25, min_class_size = 10,
                 datasets = None, metric = "auroc", cc_root = None):
        """Initialize the TargetMate class

        Args:
            models_path(str): Directorty where models will be stored.
            tmp_path(str): Directory where temporary data will be stored (relevant at predict time) (default=None)
            base_clf(clf): Classifier instance, containing fit and predict_proba methods (default="tpot")
                The following strings are also accepted: "logistic_regression", "random_forest", "naive_bayes" and "tpot"
                By default, sklearn LogisticRegressionCV is used.
            cv(int): Number of cv folds. The default cv generator used is Stratified K-Folds (default=5).
            k(int): Number of molecules to look across when doing the applicability domain (default=5).
            min_sim(float): Minimum Tanimoto chemical similarity to consider in the applicability domain determination (default=0.3).
            min_class_size(int): Minimum class size acceptable to train the classifier (default=10).
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign3 predictor are used.
            metric(str): Metric to use in the meta-prediction (auroc or aupr) (default="auroc").
            cc_root(str): CC root folder (default=None).
        """
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            raise Exception("Specified models directory does not exist: %s" % self.models_path)
        # Temporary path
        if not tmp_path:
            import uuid
            self.tmp_path = os.path.join(Config().PATH.CC_TMP, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        if not os.path.exists(tmp_path):
            raise Exception("Specified temporary directory does not exist: %s" % self.tmp_path)
        # Set the base classifier
        if type(base_clf) == str:
            if base_clf == "logistic_regression":
                from sklearn.linear_model import LogisticRegressionCV
                self.base_clf = LogisticRegressionCV(cv = 3, class_weight = "balanced", max_iter = 100)
            if base_clf == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.base_clf = RandomForestClassifier(n_estimators = 100, class_weight = "balanced", n_jobs = 1)
            if base_clf == "naive_bayes":
                from sklearn.naive_bayes import BernoulliNB
                self.base_clf = BernoulliNB()
            if base_clf == "tpot":
                from tpot import TPOTClassifier # @mbertoni: install TPOT please in the Chemical Checker...
                self.base_clf = TPOTClassifier(generations = 100, scoring = "roc_auc",
                                     config_dict = "TPOT light", verbosity = 0, n_jobs = 1)
                #self.base_clf = TPOTClassifier(generations = 100, scoring = "roc_auc",
                #                               verbosity = 0, n_jobs = 1)
                self._is_tpot = True
            else:
                self._is_tpot = False
        else:
            self.base_clf = base_clf
        # Crossvalidation to determine the performances of the individual predictors
        self.cv = cv
        # K-neighbors to search
        self.k = k
        # Minimal chemical similarity to consider
        self.min_sim = min_sim
        # Minimum size of the minority class
        self.min_class_size = min_class_size
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Store the paths to the sign3 (only the ones that have been already trained)
        if not datasets:
            self.datasets = ["%s%d.001" % (x, i) for x in ["A","B","C","D","E"] for i in [1,2,3,4,5]]
        else:
            self.datasets = datasets
        # Metric to use
        self.metric = metric
        # Others
        self._is_fitted  = False
        self._is_trained = False

    def _read_sign3(self, dataset, idxs = None, sign_folder = None, is_prd = False):
        # Identify HDF5 file
        if not sign_folder:
            if is_prd:
                h5file = os.path.join(self.tmp_path, dataset)
            else:
                h5file = os.path.join(self.models_path, dataset)
        else:
            h5file = os.path.join(sign_folder, dataset)        
        # Read the file
        with h5py.File(h5file, "r") as hf:
            if not idxs:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V

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
        variance = np.average((values-average)**2, weights=weights)
        return (average, math.sqrt(variance))

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
        # AUROC
        auroc = metrics.roc_auc_score(yt, yp)
        auroc_w = (max(auroc, 0.5) - 0.5) / (1. - 0.5)
        perfs["auroc"] = (auroc, auroc_w + 1e-6)
        # AUPR
        prec, rec, _ = metrics.precision_recall_curve(yt, yp)
        aupr = metrics.auc(rec, prec)
        aupr_w = (aupr - 0.) / (1. - 0.)
        perfs["aupr"] = (aupr, aupr_w + 1e-6)
        return perfs

    def metapredict(self, yp_dict, perfs):
        """Do meta-prediciton based on dataset-specific predictions.
        Weights are given according to the performance of the individual predictors.
        Standard deviation across predictions is kept to estimate applicability domain.
        """
        M = []
        w = []
        for dataset in self.datasets:
            w += [perfs[dataset]["perf_test"][self.metric][1]] # Get the weight
            M += [yp_dict[dataset]]
        M = np.array(M)
        w = np.array(w)
        prds = []
        stds = []
        for j in xrange(M.shape[1]):
            avg, std = self.avg_and_std(M[:,j], w)
            prds += [avg]
            stds += [std]
        return np.clip(prds, 0.001, 0.999), np.clip(stds, 0.001, None)

    def fingerprint_arena(self, smiles, use_checkpoints = False, is_prd = False):
        """Read smiles string"""
        if is_prd:
            smi_file = os.path.join(self.tmp_path, "arena.smi")
            fps_file = os.path.join(self.tmp_path, "arena.fps")
        else:
            smi_file = os.path.join(self.models_path, "arena.smi")
            fps_file = os.path.join(self.models_path, "arena.fps")
        if not use_checkpoints or not os.path.exists(fps_file):
            self.__log.debug("Writing SMILES")
            with open(smi_file, "w") as f:
                for i, smi in enumerate(smiles):
                    f.write("%s\t%i\n" % (smi, i))
            self.__log.debug("Writing Fingerprints")
            reader = chemfp.read_molecule_fingerprints(type = "RDKit-Morgan", source = smi_file, format = "smi")
            writer = chemfp.open_fingerprint_writer(fps_file, reader.metadata)
            writer.write_fingerprints(reader)
            writer.close()
            reader.close()
        arena = chemfp.load_fingerprints(fps_file)
        return arena

    @staticmethod
    def calculate_bias(yt, yp):
        """Calculate the bias of the QSAR model"""
        return np.array([np.abs(t-p) for t,p in zip(yt, yp)])
    
    @staticmethod
    def read_smiles(smi, standardize):
        try:
            mol = Chem.MolFromSmiles(smi)
            if standardize:
                mol = standardise.run(mol)
        except:
            return None
        ik  = Chem.rdinchi.InchiToInchiKey(Chem.rdinchi.MolToInchi(mol)[0])
        smi = Chem.MolToSmiles(mol)
        return ik, smi

    def knearest_search(self, query_fps, target_fps, N = None):
        """Nearest neighbors search using chemical fingerprints"""
        if target_fps is None:
            k = np.min([self.k, len(query_fps)-1])
            neighs = chemfp.knearest_tanimoto_search_symmetric(query_fps, k = k, threshold = self.min_sim)
        else:
            k = np.min([self.k, len(target_fps)])
            neighs = chemfp.knearest_tanimoto_search(query_fps, target_fps, k = k, threshold = self.min_sim)
        if N is None: N = len(query_fps)
        sims = np.zeros((N, k), dtype = np.float32)
        idxs = np.zeros((N, k), dtype = np.int)
        for query_id, hits in neighs:
            q_idx = int(query_id)
            hits  = hits.get_ids_and_scores()
            sims_ = []
            idxs_ = []
            for h in hits:
                sims_ += [h[1]]
                idxs_ += [int(h[0])]
            sims[q_idx,:len(sims_)] = sims_
            idxs[q_idx,:len(idxs_)] = idxs_
        return sims, idxs
 
    def calculate_weights(self, query_fps, target_fps, stds, bias, N = None):
        """Calculate weights using adaptation of Aniceto et al 2016."""
        self.__log.debug("Finding nearest neighbors")
        sims, idxs = self.knearest_search(query_fps, target_fps, N)
        self.__log.debug("Calculating weights from std and bias")
        weights = []
        for i, idxs_ in enumerate(idxs):
            ws = sims[i]/(stds[idxs_]*bias[idxs_])
            weights += [np.max(ws)]
        return np.array(weights)

    def fit(self, data, standardize = True, use_checkpoints = False):
        """
        Fit SMILES-activity data.
        Invalid SMILES are removed from the prediction.

        Args:
            data(str or list of tuples): 
            standardize(bool): If True, SMILES strings will be standardized (default=True)
            use_checkpoints(bool): Store signature files and others that can be reutilized (default=False)
        """
        if not use_checkpoints:
            # Cleaning models directory
            self.__log.debug("Cleaning previous checkpoints")
            shutil.rmtree(self.models_path)
            os.mkdir(self.models_path)
        # Read data
        self.__log.info("Reading data")
        # Read data if it is a file
        if type(data) == str:
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter = "\t"):
                    data += [(r[0], int(r[1]))]
        # Get only valid SMILES strings
        self.__log.info("Parsing SMILES strings, keeping only valid ones for training.")
        data_ = []
        for i, d in enumerate(data):
            m = self.read_smiles(d[0], standardize)
            if not m: continue
            data_ += [(i, m[0], m[1], int(d[1]))]
        data = data_
        y      = np.array([d[-1] for d in data])
        smiles = np.array([d[2] for d in data])
        # Save training data
        with open(self.models_path + "/trained_data.pkl", "w") as f:
            pickle.dump(data, f)
        # Check that there are enough molecules for training.
        ny = np.sum(y)
        if ny < self.min_class_size or (len(y) - ny) < self.min_class_size:
            self.__log.warning("Not enough valid molecules in the minority class... Just keeping training data")
            self._is_fitted = True
            return
        # Get signatures
        self.__log.info("Calculating sign3 for every molecule.")
        for dataset in self.datasets:
            destination_dir = os.path.join(self.models_path, dataset)
            if os.path.exists(destination_dir) and use_checkpoints:
                continue
            else:
                self.__log.debug("Calculating sign3 for %s" % dataset)
                s3 = self.cc.get_signature("sign3", "full", dataset)
                s3.predict_from_smiles([d[2] for d in data], destination_dir)
        # Fitting the global predictor
        self.__log.info("Fitting individual classifiers trained on full data")
        clf_ensemble = []
        if self._is_tpot:
            pipes = []
        for dataset in self.datasets:
            self.__log.debug("Working on %s" % dataset)
            clf = clone(self.base_clf)
            X = self._read_sign3(dataset)
            clf.fit(X, y)
            if self._is_tpot:
                pipes += [clf.fitted_pipeline_]
            clf_ensemble += [(dataset, clf)]
        # Initialize cross-validation generator
        skf = StratifiedKFold(n_splits=np.min([self.cv, ny]), shuffle=True, random_state=42)
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
                # Fit the classifier
                if self._is_tpot:
                    clf = pipes[i]
                else:
                    clf = clone(self.base_clf)
                X_train = self._read_sign3(dataset, train_idx)
                y_train = y[train_idx]
                clf.fit(X_train, y_train)
                # Make predictions on train set itself
                yps_train[dataset] += [p[1] for p in clf.predict_proba(X_train)]
                # Make predictions on test set itself
                X_test = self._read_sign3(dataset, test_idx)
                y_test = y[test_idx]
                yps_test[dataset] += [p[1] for p in clf.predict_proba(X_test)]
            yts_train += list(y_train)
            yts_test  += list(y_test)
            smi_test  += list(smiles[test_idx])
        # Evaluate individual performances
        self.__log.info("Evaluating dataset-specific performances based on the CV and getting weights correspondingly")
        perfs = {}
        for dataset in self.datasets:
            ptrain = self.performances(yts_train, yps_train[dataset])
            ptest  = self.performances(yts_test, yps_test[dataset])
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
        ptest  = self.performances(yts_test, mps_test)
        perfs["MetaPred"] = {"perf_train": ptrain, "perf_test": ptest}
        # Save performances
        with open(self.models_path + "/perfs.json", "w") as f:
            json.dump(perfs, f)
        # Save ensemble
        with open(self.models_path + "/ensemble.pkl", "w") as f:
            pickle.dump(clf_ensemble, f)
        # Nearest neighbors
        self.__log.info("Calculating nearest-neighbors model to be used in the applicability domain.")
        self.__log.debug("Getting fingerprint arena")
        fps_test = self.fingerprint_arena(smi_test, use_checkpoints = use_checkpoints, is_prd = False)
        # Save AD data
        self.__log.info("Calculating applicability domain weights")
        self.__log.debug("Working on the bias")
        bias_test    = self.calculate_bias(yts_test, mps_test)
        self.__log.debug("Working on the weights")
        weights_test = self.calculate_weights(fps_test, None, std_test, bias_test)
        self.__log.debug("Stacking AD data")
        ad_data      = np.vstack((std_test, bias_test, weights_test)).T
        self.__log.info("Saving applicability domain weights")
        with open(self.models_path + "/ad_data.pkl", "w") as f:
            pickle.dump(ad_data, f)
        # Cleaning up, if necessary
        if not use_checkpoints:
            self.__log.debug("Removing signature files")
            for dataset in datasets:
                os.remove(os.path.join(self.models_path, dataset))
        # Finish
        self._is_fitted  = True
        self._is_trained = True

    def predict(self, data, datasets = None, standardize = True, known = True):
        '''
        Predict SMILES-activity data.
        Invalid SMILES are given no prediction.
        We provide the ouptut probability, the applicability domain and the precision of the model (1 - std).

        Args:
            data(str or list): SMILES strings expressed as a list or a file.
                If a folder name is given, then TargetMate assumes that signatures are provided (use with caution).
            datasets(list): Subset of datasets to use in the metaprediction. All by default (default=None) 
            standardize(bool): If True, SMILES strings will be standardized (default=True)
            known(bool): Look for exact matches based on InChIKey (default=True)
        '''
        if not self._is_fitted:
            raise Exception("TargetMate instance needs to be fitted first")
        # Dataset subset
        if not datasets:
            my_datasets = set(self.datasets)
        else:
            my_datasets = set(datasets)
        if len(my_datasets.intersection(self.datasets)) < 1:
            raise Exception("At least one valid dataset is necessary")
        sign_folder = None
        if type(data) == str:
            data = os.path.abspath(data)
            if os.path.isfolder(data):
                # When data is a folder, we assume it contains signatures
                self.__log.debug("Signature folder found")
                sign_folder = os.path.abspath(data)
                # We need to get the SMILES strings from these signatures, and make sure everything is in the same order.
                data = []
                # @mduran: Do this.
            elif os.path.ifile(data):
                self.__log.info("Reading data from file")
                # Read data if it is a file
                if type(data) == str:
                    if not os.path.isfile(data):
                        raise Exception("File %d does not exist" % data)
                    with open(data, "r") as f:
                        data = []
                        for r in csv.reader(f, delimiter = "\t"):
                            data += [r[0]]
            else:
                raise Exception("%s does not exist" % data)
        if not sign_folder:
            # If signatures were not provided, then we work with the temporary directory
            if os.path.exists(self.tmp_path): shutil.rmtree(self.tmp_path) 
            os.mkdir(self.tmp_path)
            # Get only valid SMILES strings
            self.__log.info("Parsing SMILES strings, keeping only valid ones for training.")
            data_ = []
            N = len(data)
            for i, d in enumerate(data):
                m = self.read_smiles(d, standardize)
                if not m: continue
                data_ += [(i, m[0], m[1])]
            data = data_
            # Get signatures
            self.__log.info("Calculating sign3 for every molecule.")
            for dataset in self.datasets:
                if dataset not in my_datasets: continue
                destination_dir = os.path.join(self.tmp_path, dataset)
                if os.path.exists(destination_dir): os.remove(destination_dir)
                self.__log.debug("Calculating sign3 for %s" % dataset)
                s3 = self.cc.get_signature("sign3", "full", dataset)
                s3.predict_from_smiles([d[2] for d in data], destination_dir)
        # Read ensemble of models
        self.__log.debug("Reading ensemble of models")
        with open(self.models_path + "/ensemble.pkl", "r") as f:
            clf_ensemble = pickle.load(f)
        # Make predictions
        self.__log.info("Doing individual predictions")
        yps  = {}
        for dataset, clf in clf_ensemble:
            if dataset not in my_datasets: continue
            X = self._read_sign3(dataset, sign_folder = sign_folder, is_prd = True)
            yps[dataset] = [p[1] for p in clf.predict_proba(X)]
        # Read performances
        self.__log.debug("Reading performances")
        with open(self.models_path + "/perfs.json", "r") as f:
            perfs = json.load(f)
        # Do the metaprediction
        self.__log.info("Metaprediction")
        mps, std = self.metapredict(yps, perfs)
        # Do applicability domain
        self.__log.info("Calculating applicability domain")
        # Nearest neighbors
        self.__log.debug("Loading fit-time fingerprint arena")
        fps_fit = chemfp.load_fingerprints(os.path.join(self.models_path, "arena.fps"))
        # Applicability domain data
        self.__log.debug("Loading applicability domain data")
        with open(self.models_path + "/ad_data.pkl", "r") as f:
            ad_data = pickle.load(f)
        # Calculate weights
        self.__log.debug("Calculating weights")
        fps     = self.fingerprint_arena([d[2] for d in data], is_prd = True)
        weights = self.calculate_weights(fps, fps_fit, ad_data[:,0], ad_data[:,1], N)
        # Get percentiles
        self.__log.debug("Calculating percentiles of the weight (i.e. AD)")
        train_weights = ad_data[:,2]
        ad = np.array([percentileofscore(train_weights, w) for w in weights])/100.
        # Over-write with the actual value if known is True. This just uses the InChIKey to match.
        if known:
            self.__log.info("Overwriting with known data")
            with open(self.models_path + "/trained_data.pkl", "r") as f:
                tr_data = pickle.load(f)
                tr_iks = [d[1] for d in tr_data]
                tr_iks_set = set(tr_iks)
                for i, d in enumerate(data):
                    if d[1] not in tr_iks_set: continue
                    idx = tr_iks.index(d[1])
                    mps[i] = tr_data[idx][-1]
                    std[i] = 0
                    ad[i]  = 1
        self.__log.debug("Done. Mapping results to original data")
        # Map results to original data and return
        mps_ = np.full(N, np.nan)
        ad_  = np.full(N, np.nan)
        prc_ = np.full(N, np.nan)
        for i, d in enumerate(data):
            idx = d[0]
            mps_[idx] = mps[i]
            ad_[idx]  = ad[i]
            prc_[idx] = 1-std[i]
        return mps_, ad_, prc_

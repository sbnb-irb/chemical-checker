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
import os
import json
import collections
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.core.sign3 import sign3 # TO-DO: We might want to remove this after debugging!
from standardiser import standardise
from rdkit import Chem
import math
from scipy.stats import percentileofscore

@logged
class TargetMate:
    """TargetMate class"""

    def __init__(self, models_path, base_clf = None, cv = 5, k = 3, datasets = None, metric = "auroc", cc_root = None):
        """Initialize the TargetMate class

        Args:
            models_path(str): Directorty where models will be stored. 
            base_clf(clf): Classifier instance, containing fit and predict_proba methods (default=None)
                By default, sklearn LogisticRegressionCV is used.
            cv(int): Number of cv folds. The default cv generator used is Stratified K-Folds (default=5).
            k(int): Number of molecules to look across when doing the applicability domain (default=5).
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign3 predictor are used.
            metric(str): Metric to use in the meta-prediction (auroc or aupr) (default="auroc").
            cc_root(str): CC root folder (default=None).
        """
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            raise Exception("Specified models directory does not exist: %s" % self.models_path)
        # Set the base classifier
        if not base_clf:
            from sklearn.linear_model import LogisticRegressionCV
            self.base_clf = LogisticRegressionCV(cv = 3, class_weight = "balanced", max_iter = 100)
        else:
            self.base_clf = base_clf
        # Crossvalidation to determine the performances of the individual predictors
        self.cv = cv
        # K-neighbors to search
        self.k = k
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Store the paths to the sign3 (only the ones that have been already trained)
        if not datasets:
            self.datasets = ["%s%d.001" % (x, i) for x in ["A","B","C","D","E"] for i in [1,2,3,4,5] if ("%s%d" % (x, i)) != "A1"] # TO-DO: Martino, is there a way to just check how many signatures are available?
        else:
            self.datasets = datasets
        # Metric to use
        self.metric = metric            

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
        # Fast and numerically precise:
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

    @staticmethod
    def calculate_bias(yt, yp):
        """Calculate the bias of the QSAR model"""
        return np.array([np.abs(t-p) for t,p in zip(yt, yp)])

    @staticmethod
    def fpmatrix(smiles):
        from rdkit.Chem import AllChem
        from rdkit import Chem
        nBits  = 2048
        radius = 2
        V = np.zeros((len(smiles), nBits), dtype = np.int8)
        for i, smi in enumerate(smiles):
            m = Chem.MolFromSmiles(smi)
            V[i,:] = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        return V
    
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

    @staticmethod
    def calculate_weights(kneigh, fps, stds, bias):
        sims, idxs = kneigh.kneighbors(fps)
        sims = 1 - sims
        weights = []
        for i, idxs_ in enumerate(idxs):
            ws = sims[i]/(stds[idxs_]*bias[idxs_]) # Adapted from Aniceto et al. 2016. and PidginV3
            weights += [np.max(ws)]
        return np.array(weights)

    def fit(self, data, standardize = True):
        """
        Fit SMILES-activity data.
        Invalid SMILES are removed from the prediction.

        Args:
            data(str or list of tuples): 
            standardize(bool): If True, SMILES strings will be standardized.
        """
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
        # Get signatures
        self.__log.info("Calculating sign3 for every molecule.")
        s3s = {}
        for dataset in self.datasets:
            self.__log.debug("Calculating sign3 for %s" % dataset)
            s3 = self.cc.get_signature("sign3", "full", dataset)
            destination_dir = os.path.join(self.models_path, dataset)
            if os.path.exists(destination_dir):
                s3s[dataset] = sign3(destination_dir, dataset)
            else:
                s3s[dataset] = s3.predict_from_smiles([d[2] for d in data], destination_dir)
        # Initialize cross-validation generator
        y = np.array([d[-1] for d in data])
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        # Do the individual predictors
        self.__log.info("Training individual predictors with cross-validation")
        yps_train = collections.defaultdict(list)
        yps_test  = collections.defaultdict(list)
        yts_train = []
        yts_test  = []
        smiles    = np.array([d[2] for d in data])
        smi_test  = []
        for train_idx, test_idx in skf.split(smiles, y):
            self.__log.debug("CV fold")
            for dataset in self.datasets:
                destination_dir = os.path.join(self.models_path, dataset)
                s3 = s3s[dataset]
                # Fit the classifier
                clf = clone(self.base_clf)
                X_train = s3[:][train_idx]
                y_train = y[train_idx]
                clf.fit(X_train, y_train)
                # Make predictions on train set itself
                yps_train[dataset] += [p[1] for p in clf.predict_proba(X_train)]
                # Make predictions on test set itself
                X_test = s3[:][test_idx]
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
        return 
        # Fit final predictors (trained on full data) and save them
        self.__log.info("Fitting full final classifiers")
        clf_ensemble = []
        for dataset in self.datasets:
            self.__log.debug("Working on %s" % dataset)
            destination_dir = os.path.join(self.models_path, dataset)
            s3 = s3s[dataset]
            # Fit the classifier
            clf = clone(self.base_clf)
            X = s3[:]
            clf.fit(X, y)
            clf_ensemble += [(dataset, clf)]
        # Save ensemble
        with open(self.models_path + "/ensemble.pkl", "w") as f:
            pickle.dump(clf_ensemble, f)
        # Save training data
        with open(self.models_path + "/trained_data.pkl", "w") as f:
            pickle.dump(data, f)
        # Nearest neighbors
        self.__log.info("Calculating nearest-neighbors model to be used in the applicability domain.")
        kneigh = NearestNeighbors(n_neighbors=self.k, metric="jaccard")
        fps_test = self.fpmatrix(smi_test)
        kneigh.fit(fps_test)
        self.__log.info("Saving nearest-neighbors model")
        with open(self.models_path + "/kneigh.pkl", "w") as f:
            pickle.dump(kneigh, f)
        # Save AD data
        self.__log.info("Calculating applicability domain weights")
        self.__log.debug("Working on the bias")
        bias_test    = self.calculate_bias(yts_test, mps_test)
        self.__log.debug("Working on the weights")
        weights_test = self.calculate_weights(kneigh, fps_test, std_test, bias_test) ### TO-DO: FPS TRAIN!!
        self.__log.debug("Stacking AD data")
        ad_data      = np.vstack((std_test, bias_test, weights_test)).T
        self.__log.info("Saving applicability domain weights")
        with open(self.models_path + "/ad_data.pkl", "w") as f:
            pickle.dump(ad_data, f)
        
    def predict(self, data, standardize = True, known = True):
        '''
        Predict SMILES-activity data.
        Invalid SMILES are given no prediction.
        We provide the ouptut probability, the applicability domain and the precision of the model (1 - std).

        Args:
            data(str or list): SMILES strings 
            standardize(bool): If True, SMILES strings will be standardized (default=True).
            known(bool): Look for exact matches based on InChIKey (default=True).
        '''
        self.__log.info("Reading data")
        # Read data if it is a file
        if type(data) == str:
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter = "\t"):
                    data += [r[0]]
        # Get only valid SMILES strings
        self.__log.info("Parsing SMILES strings, keeping only valid ones for training.")
        data_ = []
        N = len(data)
        for i, d in enumerate(data):
            m = self.read_smiles(d[0], standardize)
            if not m: continue
            data_ += [(i, m[0], m[1])]
        data = data_
        # Get signatures
        self.__log.info("Calculating sign3 for every molecule.")
        s3s = {}
        for dataset in self.datasets:
            self.__log.debug("Calculating sign3 for %s" % dataset)
            s3 = self.cc.get_signature("sign3", "full", dataset)
            destination_dir = os.path.join(self.models_path, dataset + "_test")
            if os.path.exists(destination_dir): # TO-DO: This is useless here, remove after debugging!
                s3s[dataset] = sign3(destination_dir, dataset)
            else:
                s3s[dataset] = s3.predict_from_smiles([d[2] for d in data], destination_dir)
        # Read ensemble of models
        self.__log.debug("Reading ensemble of models")
        with open(self.models_path + "/ensemble.pkl", "r") as f:
            clf_ensemble = pickle.load(f)
        # Make predictions
        self.__log.info("Doing individual predictions")
        yps  = {}
        for dataset, clf in clf_ensemble:
            X = s3s[dataset][:]
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
        self.__log.debug("Loading nearest neighbors")
        with open(self.models_path + "/kneigh.pkl", "r") as f:
            kneigh = pickle.load(f) ## TO-DO: NN model on full dataset!!!
        # Applicability domain data
        self.__log.debug("Loading applicability domain data")
        with open(self.models_path + "/ad_data.pkl", "r") as f:
            ad_data = pickle.load(f)
        # Calculate weights
        self.__log.debug("Calculating weights")
        fps = "x" ## GET FINGERPRINTS
        weights = self.calculate_weights(kneigh, fps, ad_data[:,0], ad_data[:,1])
        # Get percentiles
        train_weights = ad_data[:,2]
        ad = np.array([percentileofscore(w) for w in weights]/100)
        # Over-write with the actual value if known is True
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
        self.__log.debut("Done!")
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

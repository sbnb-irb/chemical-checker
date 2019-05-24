"""TargetMate Classifier.

    An ensemble-based classifier based on CC signatures of different types.
    A base classifier is specified, and predictions are made for each dataset
    individually. A meta-prediction is then provided based on individual predictions,
    together with a measure of confidence for each prediction.
    In the predictions, known data is provided as 1/0 predictions. The rest of probabilities are clipped between 0.001 and 0.999.
    The classifier is greatly inspired by PidginV3: https://pidginv3.readthedocs.io/en/latest/usage/index.html
"""
import os
import json
from sklearn.model_selection import check_cv
from sklearn import metrics
from sklearn.base import clone
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import logged
from chemicalchecker.util import Config

# Utils

def avg_and_std(values, weights=None):
    """Return the (weighted) average and standard deviation.

    Args:
        values(list or array): 1-d list or array of values
        weights(list or array): By default, no weightening is applied
    """
    if weights is None:
        weights = np.ones(len(values))
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

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
    
# Main class

@logged
class TargetMate:
    """TargetMate class"""

    def __init__(self, base_clf = None, cv = 5, datasets = None, models_path = None):
        """Initialize the TargetMate class

        Args:
            base_clf(clf): Classifier instance, containing fit and predict_proba methods (default=None)
                By default, sklearn LogisticRegressionCV is used.
            cv(int or cross-validation generator): The default cv generator used is Stratified K-Folds.
                If an integer is provided, then it is the number of folds used.
                See the module `sklearn.model_selection` module for the
                list of possible cross-validation objects.
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign3 predictor are used.
            models_path(str): Directorty where models will be stored. 
        """
        # Set the base classifier
        if not base_clf:
            from sklearn.linear_model import LogisticRegressionCV
            self.base_clf = LogisticRegressionCV(class_weight = "balanced")
        else:
            self.base_clf = base_clf
        # Crossvalidation to determine the performances of the individual predictors
        self.cv = cv
        # Store the paths to the sign3 (only the ones that have been already trained)
        if not datasets:
            self.datasets = ["%s%d.001" for x in ["A","B","C","D","E"] for i in [1,2,3,4,5]] # TO-DO: Martino, is there a way to just check how many signatures are available?
        else:
            self.datasets = datasets
        # XXX
        if models_path is None:
            self.models_path = "./" # TO-DO: Martino, can you please help me with the default here?
        else:
            self.models_path = os.path.abspath(models_path)
        # XXX
        self.datasets
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker() # TO-DO: Martino, can you please help me with the arguments here?
        
    def fit(self, data, standardize = True):
        """
        Fit SMILES-activity data.
        Invalid SMILES are removed from the prediction.

        Args:
            data(str or list of tuples): 
            standardize(bool): If True, SMILES strings will be standardized.
        """
        # Read data if it is a file
        if type(data) == str:
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter = "\t"):
                    data += [(r[0], int(r[1]))]
        smiles = [d[0] for d in data]
        y = [d[1] for d in data]
        # Get signatures
        self.__log.info("Calculating sign3 for every molecule. Invalid SMILES will be removed.")
        for dataset in datasets:
            self.__log.debug("Calculating sign3 for %s" % dataset)
            s3 = cc.get_signature("sign3", "full_map", dataset) # TO-DO: martino, what does "full_map" mean?
            destination_dir = os.path.join(self.models_path, dataset)
            s3.predict_from_smiles([d[0] for d in data], destination_dir, self.standardize) # TO-DO
        # Initialize cross-validation generator
        cv = check_cv(self.cv, y, classifier = True)
        folds = list(cv.split(smiles, y))
        # Do the individual predictors
        self.__log.info("Training individual predictors with cross-validation")
        yps_train = collections.defaultdict(list)
        yps_test  = collections.defaultdict(list)
        yts_train = []
        yts_test  = []        
        i = 0
        for smi_train, y_train, smi_test, y_test in folds:
            for dataset in datasets:
                self.__log.debug("CV fold %d, dataset %s" % (i, dataset))
                destination_dir = os.path.join(self.models_path, dataset)
                s3 = cc.get_signature("sign3", destination_dir) # TO-DO: Help with this.
                # Fit the classifier
                clf = clone(self.clf)
                X_train = s3.get_values(smi_train) # TO-DO: martino, we may have mapping problems here, due to invalid smiles.
                clf.fit(X_train, y_train)
                # Make predictions on train set itself
                yps_train[dataset] += [p for p in clf.predict(X_train)]
                # Make predictions on test set itself
                X_test = s3.get_values(smi_test) 
                yps_test[dataset] += [p for p in clf.predict(X_test)]
            yts_train += list(y_train)
            yts_test += list(y_test) 
            i += 1
        # Evaluate individual performances
        self.__log.info("Evaluating dataset-specific performances based on the CV and getting weights correspondingly.")
        perfs = [] 
        for dataset in datasets:
            ptrain = performances(yts_train, yps_train[dataset])
            ptest  = performances(yts_test, yts_test[dataset])
            perfs += [{"dataset": dataset, "perf_train": ptrain, "perf_test": ptest}]
        # Meta-predictor on train and test data
        self.__log.info("Meta-predictions on train and test data")
        self.__log.debug("Assembling for train set")
        mp_train = []
        for dataset in datasets:
            mp_train += [yps_train[dataset]]
        self.__log.debug("Assembling for test set")
        mp_test = []
        for dataset in datasets:
            mp_test += [yps_test[dataset]]
        
        with open(self.models_path + "/individual_perfs.json", "w") as f:
            json.dump(perfs, f)

        # Fit final predictors (trained on full data) and save them
        self.__log.info("Fitting full final classifiers")
        for dataset in datasets:
            self.__log.debug("Working on ")
            destination_dir = os.path.join(self.models_path, dataset)
        # Nearest neighbors

    def predict(self, data):
        "XXX"
        

    def predict_proba(self, data):
        "XXX"

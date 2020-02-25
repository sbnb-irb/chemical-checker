"""Remove outliers"""
import os
import joblib
from sklearn.ensemble import IsolationForest

from chemicalchecker.util import logged

@logged
class OutlierRemover(object):
    """Remove outliers"""

    def __init__(self, sign1, n_estimators=100, n_jobs=-1):
        """Initialize the outlier remover"""
        self.sign = sign1
        if self.sign.cctype != "sign1":
            raise Exception("Outliers are only removed in signature 1")
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.model_path = os.path.join(self.sign.model_path, "outlier_removal.pkl")

    def fit(self):
        """Fit the outlier remover"""
        self.__log.debug("Fitting an isolation forest")
        X = sign1[:]
        model = IsolationForest(n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        model.fit(X)
        joblib.dump(model, self.model_path)

    def predict(self, sign1):
        """Predict outliers"""
        model = joblib.load(self.model_path)
        X = ""
        model.predict()

"""Remove outliers"""
import os
import joblib
from sklearn.ensemble import IsolationForest

from .base import BaseTransform

class OutlierRemover(object):
    """Remove outliers"""
    def __init__(self, sign1, max_keys=100000, n_estimators=100, n_jobs=4):
        """Initialize the outlier remover"""
        BaseTransform.__init__(self, sign1, "outliers", max_keys)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self):
        """Fit the outlier remover"""
        V = self.subsample()[0]
        mod = IsolationForest(n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        mod.fit(X)
        self.model_path = os.path.join(self.model_path, self.name+".joblib")
        joblib.dump(mod, self.model_path)
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        """Predict outliers"""
        mod = joblib.load(self.model_path)
        X = ""
        model.predict()
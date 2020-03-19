"""Remove outliers"""
import os
import joblib
from sklearn.ensemble import IsolationForest

from .base import BaseTransform

class OutlierRemover(object):
    """Remove outliers"""
    def __init__(self, sign1, max_outliers, max_keys=100000, n_estimators=100, n_jobs=4):
        """Initialize the outlier remover"""
        BaseTransform.__init__(self, sign1, "outliers", max_keys)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_outliers = max_outliers

    def fit(self):
        """Fit the outlier remover"""
        V = self.subsample()[0]
        mod = IsolationForest(n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        mod.fit(X)
        self.model_path = os.path.join(self.model_path, self.name+".joblib")
        joblib.dump(mod, self.model_path)
        self.predict(self.sign_ref, self.max_outliers)
        self.predict(self.sign, self.max_outliers)
        self.save()

    def predict(self, sign1, max_outliers=None):
        """Predict outliers"""
        self.predict_check(sign1)
        if max_outliers is None: max_outliers = self.max_outliers
        mod = joblib.load(self.model_path)
        X = ""
        model.predict()


    def predict(self, sign1):
        self.predict_check(sign1)
        pca = joblib.load(os.path.join(self.model_path, self.name+".joblib"))
        V = pca.transform(sign1[:])
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)
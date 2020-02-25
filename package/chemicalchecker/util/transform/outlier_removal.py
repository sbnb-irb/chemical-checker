"""Remove outliers"""
import os
import joblib
from sklearn.ensemble import IsolationForest

from .base import BaseTransform

class OutlierRemover(object):
    """Remove outliers"""
    def __init__(self, sign1, max_keys=100000, n_estimators=100, n_jobs=-1):
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


class Pca(BaseTransform):
    """Do a PCA"""
    def __init__(self, sign1, n_components=0.9, max_keys=10000, **kwargs):
        BaseTransform.__init__(self, sign1, "pca", max_keys)
        self.n_components = n_components

    def fit(self):
        V = self.subsample()[0]
        pca = PCA(n_components = self.n_components)
        pca.fit(V)
        joblib.dump(pca, os.path.join(self.model_path, self.name+".joblib"))
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        self.predict_check(sign1)
        pca = joblib.load(os.path.join(self.model_path, self.name+".joblib"))
        V = pca.transform(sign1[:])
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)

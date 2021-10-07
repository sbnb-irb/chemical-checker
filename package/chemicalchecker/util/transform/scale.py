"""Robustly scale a dataset."""
import os
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from .base import BaseTransform


class Scale(BaseTransform):
    """Scale class."""

    def __init__(self, sign1, *args, tmp=False, percentile=99.9, z_extreme=10,
                 max_keys=10000, **kwargs):
        BaseTransform.__init__(self, sign1, "scale", max_keys, tmp)
        self.percentile = percentile
        self.z_extreme = z_extreme

    def fit(self):
        V = self.subsample()[0]
        scl = RobustScaler()
        scl.fit(V)
        joblib.dump(scl, os.path.join(self.model_path, self.name + ".joblib"))
        self.up = np.min([np.percentile(V, self.percentile), self.z_extreme])
        self.dw = np.max(
            [np.percentile(V, 100 - self.percentile), -self.z_extreme])
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        self.predict_check(sign1)
        scl = joblib.load(os.path.join(self.model_path, self.name + ".joblib"))
        V = scl.transform(sign1[:])
        V[V > self.up] = self.up
        V[V < self.dw] = self.dw
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)

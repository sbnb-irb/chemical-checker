"""Do a PCA."""
import os
import joblib
from sklearn.decomposition import PCA

from .base import BaseTransform


class Pca(BaseTransform):
    """Pca class."""

    def __init__(self, sign1, *args, tmp=False, n_components=0.9,
                 max_keys=10000, **kwargs):
        BaseTransform.__init__(self, sign1, "pca", max_keys, tmp)
        self.n_components = n_components

    def fit(self):
        V = self.subsample()[0]
        pca = PCA(n_components=self.n_components)
        pca.fit(V)
        joblib.dump(pca, os.path.join(self.model_path, self.name + ".joblib"))
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        self.predict_check(sign1)
        pca = joblib.load(os.path.join(self.model_path, self.name + ".joblib"))
        V = pca.transform(sign1[:])
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)

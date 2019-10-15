import numpy as np
import pandas as pd

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

class McpClassifier():
    """Mondrian ICP classifier"""

    def __init__(icp_clf, **kwargs):
        self.cp = IcpClassifier(ClassifierNc(ClassifierAdapter(base_clf), MarginErrFunc()))

    def fit(X, y):
        """Fit the ICP"""
        self.cp.fit(X, y)

    def calibrate(X, y):
        """Calibrate for each class independently"""

        self.labels = sorted(set(y))
        self.cps = []
        for l in labels:
            mask = y == l
            cp = clone(self.cp)
            self.cps += [cp.calibrate(X[mask], y[mask])]

    self predict(X, )

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = icp.predict(data.data[test, :], significance=0.1)
header = np.array(['c0','c1','c2','Truth'])
table = np.vstack([prediction.T, data.target[test]]).T
df = pd.DataFrame(np.vstack([header, table]))
print(df)
import pandas as pd
import numpy as np
import random

class InputData:
    """A simple input data class"""
    
    def __init__(self, data):
        """Initialize input data class"""
        self.idx       = np.array([d[0] for d in data])
        self.activity  = np.array([float(d[1]) for d in data])
        self.smiles    = np.array([d[2] for d in data])
        self.inchikey  = np.array([d[3] for d in data])

    def __iter__(self):
        for idx, v in self.as_dataframe().iterrows():
            yield v

    def __getitem__(self, idxs):
        data = InputData([])
        data.idx      = self.idx[idxs]
        data.activity = self.activity[idxs]
        data.smiles   = self.smiles[idxs]
        data.inchikey = self.inchikey[idxs]
        return data

    def as_dataframe(self):
        df = pd.DataFrame({
            "idx": self.idx,
            "activity": self.activity,
            "smiles": self.smiles,
            "inchikey": self.inchikey
            })
        return df        

    def shuffle(self, inplace=True):
        """Shuffle data"""
        ridxs = [i for i in range(0, len(self.idx))]
        random.shuffle(ridxs)
        if inplace:
            self.idx      = self.idx[ridxs]
            self.activity = self.activity[ridxs]
            self.smiles   = self.smiles[ridxs]
            self.inchikey = self.inchikey[ridxs]
        else:
            return self.__getitem__[ridxs]


class Prediction:
    """A simple prediction class"""

    def __init__(self, datasets, y_pred, is_ensemble, weights=None):
        self.is_ensemble = is_ensemble
        self.datasets = datasets
        if is_ensemble:
            self.y_pred_ens = y_pred
            self.weights = weights
            self.y_pred = self.metapredict(self.datasets)
        else:
            self.y_pred = y_pred

    def metapredict(self, datasets=None):
        """Metapredict using a double weighting scheme"""
        if not self.is_ensemble:
            return self.y_pred
        if not datasets:
            datasets = self.datasets
        else:
            datasets = sorted(set(self.datasets).intersection(datasets))
        idxs = [self.datasets.index(ds) for ds in datasets]
        if self.weights is None:
            weights = np.ones(len(idxs))
        else:
            weights = np.array([self.weights[ds] for ds in datasets])
        weights = weights / np.sum(weights)
        v = self.y_pred_ens[:,:,idxs]
        pweights = np.max(v, axis = 1)
        y_pred = np.zeros((v.shape[0], v.shape[1]))
        for i in range(0, y_pred.shape[0]):
            pw = pweights[i] / np.sum(pweights[i])
            w  = weights*pw
            w  = w / np.sum(w)
            for j in range(0, y_pred.shape[0]):
                vals = v[i,j]
                y_pred[i,j] = np.average(vals, weights = w)
        return y_pred

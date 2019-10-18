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

    def __init__(self, y_pred, weights, datasets):
        self.y_pred   = y_pred
        self.weights  = np.array(weights)
        self.datasets = datasets

    def metapredict(self, datasets=None):
        if not datasets:
            datasets = self.datasets
        else:
            datasets = sorted(set(self.datasets).intersection(datasets))
        idxs = [self.datasets.index(x) for x in datasets]
        return np.mean(self.y_pred[:, idxs], weights = self.weights[idxs], axis = 2)



class OutputData:
    """A simple output data class"""

    def __init__(self, data):
        """Initialize output data class"""
        pass
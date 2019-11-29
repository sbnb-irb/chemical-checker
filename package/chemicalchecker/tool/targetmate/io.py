import pandas as pd
import numpy as np
import random
import csv

from .utils.chemistry import read_smiles

# Utility functions

def reader(data):
    # Data is a file
    if type(data) == str:
        with open(data, "r") as f:
            for r in csv.reader(f, delimiter = "\t"):
                yield r
    # Data is a list
    else:
        for r in data:
            yield r

def read_data(data,
              smiles_idx,
              activity_idx=None,
              srcid_idx=None,
              standardize=False):
    """Read data.

    Args:
        data(str or list of tuples): 
        smiles_idx: Tuple or column index where smiles is specified.        
        activity_idx: Tuple or column index where activity is specified (default=None).
        srcid_idx: Tuple or column index where the source id is specified (default=None).
        standardize(bool): Standardize structures.
    
    Returns:
        InputData instance.
    """
    smiles   = []
    activity = []
    srcid    = []
    idx      = []
    inchikey = []

    for i, r in enumerate(reader(data)):
        smi = r[smiles_idx]
        m = read_smiles(smi, standardize)
        if not m: continue
        idx      += [i]
        smiles   += [m[1]]
        inchikey += [m[0]]
        if activity_idx is not None:
            activity += [float(r[activity_idx])]
        if srcid_idx is not None:
            srcid    += [r[srcid_idx]]        
    data = (idx, smiles, inchikey, activity, srcid)
    return InputData(data)


def reassemble_activity_sets(act, inact, putinact):
    """Reassemble activity sets, relevant when sampling from Universe"""
    data = []
    for x in list(act):
        data += [(x[1],  1, x[0], x[-1])]
    for x in list(inact):
        data += [(x[1], -1, x[0], x[-1])]
    n = np.max([x[0] for x in data]) + 1
    for i, x in enumerate(list(putinact)):
        data += [(i + n, 0, x[0], x[-1])]
    idx      = []
    smiles   = []
    inchikey = []
    activity = []
    srcid    = []
    for d in data:
        idx      += [d[0]]
        smiles   += [d[2]]
        inchikey += [d[3]]
        activity += [d[1]]
    data = (idx, smiles, inchikey, activity, srcid)
    return InputData(data)


def read_multiple_smiles(data_list, smiles_idx, standardize):
    smiles_ = set()
    for data in data_list:
        smis = []
        for r in reader(data):
            smis += [r[smiles_idx]]
        smiles_.update(smiles_)
    smiles_ = list(smiles_)
    smiles  = []
    for smi in smiles_:
        m = read_smiles(smi, standardize)
        if not m: continue
        smiles += [m[1]]
    return sorted(smiles)


# Classes
class InputData:
    """A simple input data class"""
    
    def __init__(self, data=None):
        """Initialize input data class"""
        if data:
            idx   = np.array(data[0])
            order = np.argsort(idx)
            self.idx      = idx[order]
            self.smiles   = np.array(data[1])[order]
            self.inchikey = np.array(data[2])[order]
            if data[3] == []:
                self.activity = None
            else:
                self.activity = np.array(data[3])[order]                
            if data[4] == []:
                self.srcid = None
            else:
                self.srcid = np.array(data[4])[order]

    def __iter__(self):
        for idx, v in self.as_dataframe().iterrows():
            yield v

    def __getitem__(self, idxs):
        data = InputData()
        data.idx = self.idx[idxs]
        if self.activity is not None:
            data.activity = self.activity[idxs]
        else:
            data.activity = None
        data.smiles = self.smiles[idxs]
        data.inchikey = self.inchikey[idxs]
        if self.srcid is not None:
            data.srcid = self.srcid[idxs]
        else:
            data.srcid = None
        return data

    def as_dataframe(self):
        df = pd.DataFrame({
            "idx": self.idx,
            "activity": self.activity,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "srcid": self.srcid
            })
        return df        

    def shuffle(self, inplace=True):
        """Shuffle data"""
        ridxs = [i for i in range(0, len(self.idx))]
        random.shuffle(ridxs)
        if inplace:
            self.idx = self.idx[ridxs]
            if self.activity is not None: self.activity = self.activity[ridxs]
            self.smiles = self.smiles[ridxs]
            self.inchikey = self.inchikey[ridxs]
            if self.srcid is not None: self.srcid = self.srcid[ridxs]
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
            for j in range(0, y_pred.shape[1]):
                vals = v[i,j]
                y_pred[i,j] = np.average(vals, weights = w)
        return y_pred

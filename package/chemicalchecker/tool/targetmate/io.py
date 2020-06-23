"""Input/output function utilities"""

import pandas as pd
import numpy as np
import random
import csv
import uuid
import pickle
import os

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

def filter_validity(data, valid_inchikeys, only_molecules=False):
    valid_inchikeys = set(valid_inchikeys)
    if not only_molecules:
        idx       = data[0]
        smiles    = data[1]
        inchikey  = data[2]
        activity  = data[3]
        srcid     = data[4]
        idx_      = []
        smiles_   = []
        inchikey_ = []
        activity_ = []
        srcid_    = []
        for i, ik in enumerate(inchikey):
            if ik not in valid_inchikeys: continue
            idx_      += [idx[i]]
            smiles_   += [smiles[i]]
            inchikey_ += [inchikey[i]]
            activity_ += [activity[i]]
            srcid_    += [srcid[i]]
        data = (idx_, smiles_, inchikey_, activity_, srcid_)
    else:
        smiles    = data[0]
        inchikey  = data[1]
        smiles_   = []
        inchikey_ = []
        for i, ik in enumerate(inchikey):
            if ik not in valid_inchikeys: continue
            smiles_   += [smiles[i]]
            inchikey_ += [inchikey[i]]
        data = (smiles_, inchikey_)
    return data
    
def read_data(data,
              smiles_idx=None,
              inchikey_idx=None,
              activity_idx=None,
              srcid_idx=None,
              standardize=False,
              use_inchikey=False,
              valid_inchikeys=None):
    """Read data.

    Args:
        data(str or list of tuples): 
        smiles_idx: Tuple or column index where smiles is specified (default=None).
        inchikey_idx: Column where the inchikey is present (default=None).        
        activity_idx: Tuple or column index where activity is specified (default=None).
        srcid_idx: Tuple or column index where the source id is specified (default=None).
        standardize(bool): Standardize structures.
        use_inchikey(bool): Use inchikey directly (default=False)

    Returns:
        InputData instance.
    """
    smiles   = []
    activity = []
    srcid    = []
    idx      = []
    inchikey = []
    if not use_inchikey:
        if smiles_idx is None:
            raise Exception("smiles_idx needs to be specified")
        for i, r in enumerate(reader(data)):
            smi = r[smiles_idx]
            m = read_smiles(smi, standardize)
            if not m: continue
            idx      += [i]
            smiles   += [m[1]]
            inchikey += [m[0]]
            if activity_idx is not None:
                activity += [float(r[activity_idx])]
            else:
                activity += [None]
            if srcid_idx is not None:
                srcid += [r[srcid_idx]]
            else:
                srcid += [None]
    else:
        if inchikey_idx is None:
            raise Exception("inchikey_idx needs to be specified")
        for i, r in enumerate(reader(data)):
            idx += [i]
            inchikey += [r[inchikey_idx]]
            if smiles_idx is not None:
                smiles += [r[smiles_idx]]
            else:
                smiles += [None]
            if activity_idx is not None:
                activity += [float(r[activity_idx])]
            else:
                activity += [None]
            if srcid_idx is not None:
                srcid += [r[srcid_idx]]
            else:
                srcid += [None]        
    data = (idx, smiles, inchikey, activity, srcid)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys)
    return InputData(data)


def reassemble_activity_sets(act, inact, putinact, valid_inchikeys=None):
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
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys)
    return InputData(data)


def read_smiles_from_multiple_data(data_list, smiles_idx, standardize=False, sort=True, valid_inchikeys=None, **kwargs):
    """Read smiles from multiple datasets"""
    smiles_ = set()
    for data in data_list:
        smis = []
        for r in reader(data):
            smis += [r[smiles_idx]]
        smiles_.update(smis)
    smiles_ = list(smiles_)
    smiles   = []
    inchikey = []
    for smi in smiles_:
        m = read_smiles(smi, standardize)
        if not m: continue
        smiles += [m[1]]
        inchikey += [m[0]]
    data = (smiles, inchikey)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys, only_molecules=True)
    return SmilesData(data, sort=sort)


def data_to_disk(data, tmp_dir):
    data.on_disk(tmp_dir)


def data_from_disk(data):
    if type(data) is str:
        with open(data, "rb") as f:
            data = pickle.load(f)        
        return data
    else:
        return data


# Classes
class InputData:
    """A simple input data class"""
    
    def __init__(self, data=None):
        """Initialize input data class"""
        self.tag = str(uuid.uuid4())
        if data is not None:
            idx   = np.array(data[0])
            order = np.argsort(idx)
            self.idx      = idx[order]
            self.smiles   = np.array(data[1])[order]
            self.inchikey = np.array(data[2])[order]
            if data[3] == []:
                self.activity = None
            else:
                self.activity = np.array(data[3])[order]
                try:
                    self.activity.astype(np.float)
                except:
                    raise Exception("Activities are not of numeric type!")            
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

    @staticmethod
    def sel(ary, idxs):
        if idxs is None:
            return ary
        if ary is None:
            return None
        return ary[idxs]

    def as_dict(self, idxs):
        res = {
            "idx": self.sel(self.idx, idxs),
            "activity": self.sel(self.activity, idxs),
            "smiles": self.sel(self.smiles, idxs),
            "inchikey": self.sel(self.inchikey, idxs),
            "srcid": self.sel(self.srcid, idxs)
        }
        return res

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

    def on_disk(self, tmp_dir):
        data_path = os.path.join(tmp_dir, "data_objects")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        data_path = os.path.join(data_path, self.tag)
        with open(data_path, "wb") as f:
            pickle.dump(self, f)
        return data_path


class SmilesData(object):
    """A simple smiles data container"""

    def __init__(self, data, sort):
        """Initialize"""
        smiles   = np.array(data[0])
        inchikey = np.array(data[1])
        if sort:
            order     = np.argsort(smiles)
            smiles    = smiles[order]
            inchikey  = inchikey[order]
        self.smiles   = smiles
        self.inchikey = inchikey


class Prediction(object):
    """A simple prediction class"""

    def __init__(self, datasets, y_true, y_pred, is_ensemble, weights=None):
        self.is_ensemble = is_ensemble
        self.datasets = datasets
        self.y_true = y_true
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


class Explanation(object):
    """Shapley explanation results"""
    def __init__(self, datasets, shaps, is_ensemble):
        self.is_ensemble = is_ensemble
        self.datasets = datasets
        self.shaps = shaps
